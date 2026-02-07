#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

#define WARP_SIZE 32
#define BLOCK_SIZE 128

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction using shared memory
__device__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? smem[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

__device__ float block_reduce_max(float val, float* smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? smem[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

// Each block handles one output row
// Threads cooperate on dot products and V accumulation
__global__ void waller_attention_v4(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    int head_dim,
    float scale
) {
    int row = blockIdx.x;
    if (row >= seq_len) return;
    
    int tid = threadIdx.x;
    
    extern __shared__ float smem[];
    float* q_shared = smem;                    // head_dim floats
    float* reduce_smem = smem + head_dim;      // BLOCK_SIZE/WARP_SIZE floats
    float* acc_shared = reduce_smem + 8;       // head_dim floats
    
    // Load Q row into shared memory
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        q_shared[d] = Q[row * head_dim + d];
    }
    
    // Initialize accumulator
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        acc_shared[d] = 0.0f;
    }
    __syncthreads();
    
    float m = -INFINITY;  // running max (per thread, will reduce)
    float l = 0.0f;       // running sum
    
    // Process each column (causal: col <= row)
    for (int col = 0; col <= row; col++) {
        // Parallel dot product: each thread handles part of dimensions
        float partial_dot = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            partial_dot += q_shared[d] * K[col * head_dim + d];
        }
        
        // Reduce dot product across block
        float dot = block_reduce_sum(partial_dot, reduce_smem);
        __syncthreads();
        
        // Broadcast dot to all threads
        if (tid == 0) reduce_smem[0] = dot * scale;
        __syncthreads();
        dot = reduce_smem[0];
        
        // Online softmax update (all threads have same dot now)
        float m_new = fmaxf(m, dot);
        float exp_diff = expf(m - m_new);
        float exp_dot = expf(dot - m_new);
        
        l = l * exp_diff + exp_dot;
        
        // Update accumulator: rescale old and add new V contribution
        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            acc_shared[d] = acc_shared[d] * exp_diff + exp_dot * V[col * head_dim + d];
        }
        __syncthreads();
        
        m = m_new;
    }
    
    // Normalize and write output
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        O[row * head_dim + d] = acc_shared[d] / l;
    }
}

void run_benchmark(int seq_len, int head_dim, int iters) {
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("Seq: %d | HeadDim: %d | Iters: %d\n", seq_len, head_dim, iters);
    
    size_t mat_size = (size_t)seq_len * head_dim * sizeof(float);
    
    float *h_Q = (float*)malloc(mat_size);
    float *h_K = (float*)malloc(mat_size);
    float *h_V = (float*)malloc(mat_size);
    
    srand(42);
    for (int i = 0; i < seq_len * head_dim; i++) {
        h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_K[i] = (float)rand() / RAND_MAX - 0.5f;
        h_V[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, mat_size));
    CHECK_CUDA(cudaMalloc(&d_K, mat_size));
    CHECK_CUDA(cudaMalloc(&d_V, mat_size));
    CHECK_CUDA(cudaMalloc(&d_O, mat_size));
    
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, mat_size, cudaMemcpyHostToDevice));
    
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t smem_size = (head_dim + 8 + head_dim) * sizeof(float);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        waller_attention_v4<<<seq_len, BLOCK_SIZE, smem_size>>>(d_Q, d_K, d_V, d_O, seq_len, head_dim, scale);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        waller_attention_v4<<<seq_len, BLOCK_SIZE, smem_size>>>(d_Q, d_K, d_V, d_O, seq_len, head_dim, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    
    long long flops = (long long)seq_len * (seq_len + 1) / 2 * (2 * head_dim + 4);
    double tflops = (flops / 1e12) / (avg_ms / 1000.0);
    
    size_t standard_mem = (size_t)seq_len * seq_len * sizeof(float);
    float mem_reduction = 100.0f * (1.0f - (4.0f * mat_size) / (standard_mem + 4 * mat_size));
    
    printf("Time: %.3f ms | %.2f TFLOPS | Mem Reduction: %.1f%%\n", avg_ms, tflops, mem_reduction);
    printf("Standard NxN: %.2f GB | Waller: %.4f GB\n", standard_mem / 1e9, (4.0f * mat_size) / 1e9);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER OPERATOR V4 - PARALLEL REDUCTION                      ║\n");
    printf("║     Geodesic Attention Engine (GAE)                             ║\n");
    printf("║     Copyright 2026 Eric Waller                                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s | SMs: %d\n", prop.name, prop.multiProcessorCount);
    
    run_benchmark(1024, 128, 10);
    run_benchmark(2048, 128, 10);
    run_benchmark(4096, 128, 10);
    run_benchmark(8192, 128, 5);
    run_benchmark(16384, 128, 3);
    
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("BENCHMARK COMPLETE\n");
    
    return 0;
}

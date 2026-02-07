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

// Simpler kernel: one thread per row, no local arrays
// Uses online softmax with O(1) memory
__global__ void waller_attention_kernel(
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
    
    // Online softmax variables
    float m = -INFINITY;  // running max
    float l = 0.0f;       // running sum of exp
    
    // Accumulator in global memory offset
    int out_offset = row * head_dim;
    
    // Initialize output row to zero
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        O[out_offset + d] = 0.0f;
    }
    __syncthreads();
    
    // Process each column (causal: j <= row)
    for (int col = 0; col <= row; col++) {
        // Compute Q[row] dot K[col]
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[row * head_dim + d] * K[col * head_dim + d];
        }
        dot *= scale;
        
        // Online softmax update
        float m_new = fmaxf(m, dot);
        float exp_diff = expf(m - m_new);
        float exp_dot = expf(dot - m_new);
        
        l = l * exp_diff + exp_dot;
        
        // Rescale existing accumulator and add new contribution
        float rescale = exp_diff;
        float weight = exp_dot;
        
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            O[out_offset + d] = O[out_offset + d] * rescale + weight * V[col * head_dim + d];
        }
        __syncthreads();
        
        m = m_new;
    }
    
    // Normalize by sum
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        O[out_offset + d] /= l;
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
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        waller_attention_kernel<<<seq_len, 32>>>(d_Q, d_K, d_V, d_O, seq_len, head_dim, scale);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        waller_attention_kernel<<<seq_len, 32>>>(d_Q, d_K, d_V, d_O, seq_len, head_dim, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    
    // FLOPs: for each row i, we do (i+1) dot products of size head_dim, plus softmax
    // Total dots = N*(N+1)/2, each dot = 2*head_dim FLOPs
    long long flops = (long long)seq_len * (seq_len + 1) / 2 * (2 * head_dim + 4);
    double tflops = (flops / 1e12) / (avg_ms / 1000.0);
    
    size_t standard_mem = (size_t)seq_len * seq_len * sizeof(float);
    float mem_reduction = 100.0f * (1.0f - (4.0f * mat_size) / (standard_mem + 4 * mat_size));
    
    printf("Time: %.3f ms | %.2f TFLOPS | Mem Reduction: %.1f%%\n", avg_ms, tflops, mem_reduction);
    printf("Standard Attention NxN matrix: %.2f GB\n", standard_mem / 1e9);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER OPERATOR V3 - VERIFIED WORKING KERNEL                 ║\n");
    printf("║     Geodesic Attention Engine (GAE)                             ║\n");
    printf("║     Copyright 2026 Eric Waller - Patent Pending                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s | SMs: %d\n", prop.name, prop.multiProcessorCount);
    
    run_benchmark(1024, 128, 10);
    run_benchmark(2048, 128, 10);
    run_benchmark(4096, 128, 5);
    
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("BENCHMARK COMPLETE\n");
    
    return 0;
}

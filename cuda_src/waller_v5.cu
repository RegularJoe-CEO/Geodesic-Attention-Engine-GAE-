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

#define BLOCK_M 32   // rows per block
#define BLOCK_N 32   // columns per tile  
#define BLOCK_SIZE 128

__global__ void waller_tiled_attention(
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
    float* K_tile = smem;                          // BLOCK_N * head_dim
    float* V_tile = K_tile + BLOCK_N * head_dim;   // BLOCK_N * head_dim
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    // Accumulator in registers - each thread handles subset of dimensions
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Thread handles 4 dims
    int dims_per_thread = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Iterate over column tiles
    int max_col = row + 1;  // causal
    for (int col_start = 0; col_start < max_col; col_start += BLOCK_N) {
        int col_end = min(col_start + BLOCK_N, max_col);
        int tile_cols = col_end - col_start;
        
        // Cooperatively load K and V tiles
        for (int idx = tid; idx < BLOCK_N * head_dim; idx += BLOCK_SIZE) {
            int c = idx / head_dim;
            int d = idx % head_dim;
            if (c < tile_cols) {
                K_tile[c * head_dim + d] = K[(col_start + c) * head_dim + d];
                V_tile[c * head_dim + d] = V[(col_start + c) * head_dim + d];
            }
        }
        __syncthreads();
        
        // Each thread computes partial dot products and finds tile max
        float m_tile = -INFINITY;
        float scores[32];  // Max BLOCK_N scores
        
        for (int c = 0; c < tile_cols; c++) {
            // Parallel dot product - each thread does partial
            float partial = 0.0f;
            for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
                partial += Q[row * head_dim + d] * K_tile[c * head_dim + d];
            }
            
            // Warp reduction for dot product
            for (int offset = 16; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            
            // Thread 0 of each warp has partial sum, reduce across warps
            __shared__ float warp_sums[4];
            int warp_id = tid / 32;
            int lane = tid % 32;
            if (lane == 0) warp_sums[warp_id] = partial;
            __syncthreads();
            
            float dot = 0.0f;
            if (tid == 0) {
                for (int w = 0; w < BLOCK_SIZE/32; w++) dot += warp_sums[w];
                dot *= scale;
                smem[BLOCK_N * head_dim * 2 + c] = dot;  // Store in unused smem
            }
            __syncthreads();
            
            dot = smem[BLOCK_N * head_dim * 2 + c];
            scores[c] = dot;
            m_tile = fmaxf(m_tile, dot);
        }
        
        // Online softmax update
        float m_new = fmaxf(m_i, m_tile);
        float exp_diff = expf(m_i - m_new);
        l_i *= exp_diff;
        
        // Rescale accumulator
        for (int i = 0; i < dims_per_thread; i++) {
            acc[i] *= exp_diff;
        }
        
        // Add new contributions
        for (int c = 0; c < tile_cols; c++) {
            float s = expf(scores[c] - m_new);
            l_i += s;
            for (int i = 0; i < dims_per_thread; i++) {
                int d = tid + i * BLOCK_SIZE;
                if (d < head_dim) {
                    acc[i] += s * V_tile[c * head_dim + d];
                }
            }
        }
        
        m_i = m_new;
        __syncthreads();
    }
    
    // Normalize and write output
    for (int i = 0; i < dims_per_thread; i++) {
        int d = tid + i * BLOCK_SIZE;
        if (d < head_dim) {
            O[row * head_dim + d] = acc[i] / l_i;
        }
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
    for (size_t i = 0; i < (size_t)seq_len * head_dim; i++) {
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
    
    // Shared mem: K_tile + V_tile + temp scores
    size_t smem_size = (2 * BLOCK_N * head_dim + BLOCK_N) * sizeof(float);
    
    printf("Blocks: %d | Shared mem: %.1f KB\n", seq_len, smem_size / 1024.0f);
    
    if (smem_size > 48 * 1024) {
        printf("ERROR: Shared memory too large\n");
        return;
    }
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        waller_tiled_attention<<<seq_len, BLOCK_SIZE, smem_size>>>(
            d_Q, d_K, d_V, d_O, seq_len, head_dim, scale);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        waller_tiled_attention<<<seq_len, BLOCK_SIZE, smem_size>>>(
            d_Q, d_K, d_V, d_O, seq_len, head_dim, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    
    long long flops = (long long)seq_len * (seq_len + 1) / 2 * (2 * head_dim + 4);
    double tflops = (flops / 1e12) / (avg_ms / 1000.0);
    
    size_t standard_mem = (size_t)seq_len * seq_len * sizeof(float);
    size_t waller_mem = 4 * mat_size;
    float mem_reduction = 100.0f * (1.0f - (float)waller_mem / (standard_mem + 4 * mat_size));
    
    printf("Time: %.3f ms | %.2f TFLOPS | Mem Reduction: %.1f%%\n", avg_ms, tflops, mem_reduction);
    printf("Standard NxN: %.2f GB | Waller: %.4f GB\n", standard_mem / 1e9, waller_mem / 1e9);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER OPERATOR V5 - TILED PARALLEL                          ║\n");
    printf("║     Copyright 2026 Eric Waller                                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s | SMs: %d\n", prop.name, prop.multiProcessorCount);
    
    run_benchmark(1024, 128, 10);
    run_benchmark(2048, 128, 10);
    run_benchmark(4096, 128, 10);
    run_benchmark(8192, 128, 5);
    run_benchmark(16384, 128, 3);
    
    printf("\nBENCHMARK COMPLETE\n");
    
    return 0;
}

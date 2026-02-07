// Waller Operator INT8 - TILED VERSION
// Geodesic Attention Engine (GAE)
// Copyright 2026 Eric Waller - Proprietary
// Shared memory tiling + vectorized loads

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 64
#define HEAD_DIM 64

// Tiled INT8 Triangle Kernel with Shared Memory
__global__ void waller_int8_tiled_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    float* __restrict__ Output,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    // Shared memory for K and V tiles
    __shared__ int8_t K_tile[TILE_SIZE][HEAD_DIM];
    __shared__ int8_t V_tile[TILE_SIZE][HEAD_DIM];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Load Q row into registers (stays constant)
    int8_t Q_local[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d += 16) {
        int4 q4 = *reinterpret_cast<const int4*>(&Q[row * head_dim + d]);
        *reinterpret_cast<int4*>(&Q_local[d]) = q4;
    }
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[HEAD_DIM] = {0.0f};
    
    // Process in tiles
    int max_col = row + 1;  // Triangle: only attend to col <= row
    
    for (int tile_start = 0; tile_start < max_col; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, max_col);
        int tile_len = tile_end - tile_start;
        
        // Collaborative load of K and V tiles into shared memory
        __syncthreads();
        for (int i = tid; i < tile_len; i += block_size) {
            int col = tile_start + i;
            if (col < seq_len) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 16) {
                    int4 k4 = *reinterpret_cast<const int4*>(&K[col * head_dim + d]);
                    int4 v4 = *reinterpret_cast<const int4*>(&V[col * head_dim + d]);
                    *reinterpret_cast<int4*>(&K_tile[i][d]) = k4;
                    *reinterpret_cast<int4*>(&V_tile[i][d]) = v4;
                }
            }
        }
        __syncthreads();
        
        // Process tile
        for (int i = 0; i < tile_len; i++) {
            int col = tile_start + i;
            if (col > row) break;  // Causal mask
            
            // Dot product Q_local · K_tile[i]
            int32_t dot = 0;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += (int32_t)Q_local[d] * (int32_t)K_tile[i][d];
            }
            
            float score = (float)dot * scale;
            
            // Online softmax
            float old_max = max_val;
            max_val = fmaxf(max_val, score);
            float rescale = expf(old_max - max_val);
            sum_exp = sum_exp * rescale + expf(score - max_val);
            
            float weight = expf(score - max_val);
            
            // Accumulate V
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[d] = acc[d] * rescale + weight * (float)V_tile[i][d];
            }
        }
    }
    
    // Normalize and write output
    float inv_sum = 1.0f / sum_exp;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        Output[row * head_dim + d] = acc[d] * inv_sum;
    }
}

// Larger tile version for better memory reuse
__global__ void waller_int8_tiled_large_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    float* __restrict__ Output,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    // Larger shared memory tiles (128 keys at a time)
    __shared__ int8_t K_tile[128][HEAD_DIM];
    __shared__ int8_t V_tile[128][HEAD_DIM];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Q in registers
    int8_t Q_local[HEAD_DIM];
    const int4* Q_vec = reinterpret_cast<const int4*>(&Q[row * head_dim]);
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / 16; i++) {
        *reinterpret_cast<int4*>(&Q_local[i * 16]) = Q_vec[i];
    }
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[HEAD_DIM] = {0.0f};
    
    int max_col = row + 1;
    
    for (int tile_start = 0; tile_start < max_col; tile_start += 128) {
        int tile_end = min(tile_start + 128, max_col);
        int tile_len = tile_end - tile_start;
        
        __syncthreads();
        
        // Load K/V tiles - each thread loads multiple elements
        for (int i = tid; i < tile_len; i += block_size) {
            int col = tile_start + i;
            const int4* K_vec = reinterpret_cast<const int4*>(&K[col * head_dim]);
            const int4* V_vec = reinterpret_cast<const int4*>(&V[col * head_dim]);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 16; d++) {
                *reinterpret_cast<int4*>(&K_tile[i][d * 16]) = K_vec[d];
                *reinterpret_cast<int4*>(&V_tile[i][d * 16]) = V_vec[d];
            }
        }
        __syncthreads();
        
        // Process tile from shared memory
        for (int i = 0; i < tile_len; i++) {
            int col = tile_start + i;
            if (col > row) break;
            
            int32_t dot = 0;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += (int32_t)Q_local[d] * (int32_t)K_tile[i][d];
            }
            
            float score = (float)dot * scale;
            float old_max = max_val;
            max_val = fmaxf(max_val, score);
            float rescale = expf(old_max - max_val);
            sum_exp = sum_exp * rescale + expf(score - max_val);
            float weight = expf(score - max_val);
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[d] = acc[d] * rescale + weight * (float)V_tile[i][d];
            }
        }
    }
    
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < HEAD_DIM; d++) {
        Output[row * head_dim + d] = acc[d] * inv_sum;
    }
}

void run_benchmark(int seq_len, int head_dim, int num_iterations, bool use_large) {
    printf("\n────────────────────────────────────────────────────────────────\n");
    printf("Seq: %d | Head: %d | Iters: %d | Tile: %s\n", 
           seq_len, head_dim, num_iterations, use_large ? "128" : "64");
    
    size_t int8_size = (size_t)seq_len * head_dim * sizeof(int8_t);
    size_t fp32_size = (size_t)seq_len * head_dim * sizeof(float);
    
    int8_t *h_Q = (int8_t*)malloc(int8_size);
    int8_t *h_K = (int8_t*)malloc(int8_size);
    int8_t *h_V = (int8_t*)malloc(int8_size);
    
    srand(42);
    for (size_t i = 0; i < (size_t)seq_len * head_dim; i++) {
        h_Q[i] = (int8_t)(rand() % 256 - 128);
        h_K[i] = (int8_t)(rand() % 256 - 128);
        h_V[i] = (int8_t)(rand() % 256 - 128);
    }
    
    int8_t *d_Q, *d_K, *d_V;
    float *d_Output;
    cudaMalloc(&d_Q, int8_size);
    cudaMalloc(&d_K, int8_size);
    cudaMalloc(&d_V, int8_size);
    cudaMalloc(&d_Output, fp32_size);
    
    cudaMemcpy(d_Q, h_Q, int8_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, int8_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, int8_size, cudaMemcpyHostToDevice);
    
    float scale = 1.0f / (sqrtf((float)head_dim) * 127.0f * 127.0f);
    
    int block_size = 128;
    int grid_size = (seq_len + block_size - 1) / block_size;
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        if (use_large) {
            waller_int8_tiled_large_kernel<<<grid_size, block_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, scale);
        } else {
            waller_int8_tiled_kernel<<<grid_size, block_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, scale);
        }
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        if (use_large) {
            waller_int8_tiled_large_kernel<<<grid_size, block_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, scale);
        } else {
            waller_int8_tiled_kernel<<<grid_size, block_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, scale);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ms = milliseconds / num_iterations;
    
    size_t int8_mem = 3 * int8_size + fp32_size;
    size_t standard_attn = (size_t)seq_len * seq_len * sizeof(float);
    
    double flops = (double)seq_len * ((double)seq_len + 1.0) / 2.0 * head_dim * 2.0;
    double gflops = (flops / (avg_time_ms / 1000.0)) / 1e9;
    
    float mem_reduction = 100.0f * (1.0f - (float)int8_mem / (float)standard_attn);
    
    printf("Time: %.3f ms | %.2f GFLOPS\n", avg_time_ms, gflops);
    printf("INT8 Memory: %.2f MB | Standard would need: %.2f GB\n", 
           int8_mem / 1e6, standard_attn / 1e9);
    printf("Memory Reduction: %.2f%%\n", mem_reduction);
    
    if (standard_attn > 80e9) {
        printf(">>> IMPOSSIBLE ON THIS GPU WITH STANDARD ATTENTION <<<\n");
    }
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Output);
    free(h_Q); free(h_K); free(h_V);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER INT8 TILED - TRIANGLE ENGINE                         ║\n");
    printf("║     Shared Memory Tiling + Vectorized Loads                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s\n", prop.name);
    printf("Shared Memory/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    
    // Test with 128-tile (larger tiles = better memory reuse)
    run_benchmark(8192, 64, 20, true);
    run_benchmark(16384, 64, 10, true);
    run_benchmark(32768, 64, 5, true);
    run_benchmark(65536, 64, 3, true);
    run_benchmark(131072, 64, 2, true);
    run_benchmark(262144, 64, 1, true);
    
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("TILED INT8 BENCHMARK COMPLETE\n");
    
    return 0;
}

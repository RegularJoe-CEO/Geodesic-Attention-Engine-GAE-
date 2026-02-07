// Waller Operator V2 - CUDA Implementation
// Geodesic Attention Engine (GAE)
// Copyright 2026 Eric Waller - Proprietary
// TILED PARALLEL REDUCTION - O(1) Memory with GPU Saturation

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TILE_SIZE 64
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? smem[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? smem[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return __shfl_sync(0xffffffff, val, 0);
}


// Tiled kernel - one block per row, threads cooperate on columns
__global__ void waller_operator_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Output,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int row = blockIdx.x;
    if (row >= seq_len) return;
    
    const int tid = threadIdx.x;
    
    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + TILE_SIZE * head_dim;
    float* reduce_smem = smem + 2 * TILE_SIZE * head_dim;
    
    // Load Q row into registers
    float q_reg[128];
    for (int d = 0; d < head_dim; d++) {
        q_reg[d] = Q[row * head_dim + d];
    }
    
    // Accumulators
    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float acc[128];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    
    // Process columns in tiles
    for (int tile_start = 0; tile_start <= row; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, row + 1);
        int tile_len = tile_end - tile_start;
        
        // Cooperative load K and V tiles
        for (int i = tid; i < tile_len * head_dim; i += BLOCK_SIZE) {
            int col_local = i / head_dim;
            int d = i % head_dim;
            int col_global = tile_start + col_local;
            K_tile[col_local * head_dim + d] = K[col_global * head_dim + d];
            V_tile[col_local * head_dim + d] = V[col_global * head_dim + d];
        }
        __syncthreads();
        
        // Each thread computes dot products for subset of columns
        for (int col_local = tid; col_local < tile_len; col_local += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_reg[d] * K_tile[col_local * head_dim + d];
            }
            dot *= scale;
            
            float old_max = global_max;
            global_max = fmaxf(global_max, dot);
            float rescale = expf(old_max - global_max);
            global_sum = global_sum * rescale + expf(dot - global_max);
            
            float weight = expf(dot - global_max);
            for (int d = 0; d < head_dim; d++) {
                acc[d] = acc[d] * rescale + weight * V_tile[col_local * head_dim + d];
            }
        }
        __syncthreads();
    }

    // Cross-thread reduction for global_max
    float final_max = block_reduce_max(global_max, reduce_smem);
    __syncthreads();
    
    // Rescale each thread's accumulator to final_max
    float rescale = expf(global_max - final_max);
    global_sum *= rescale;
    for (int d = 0; d < head_dim; d++) {
        acc[d] *= rescale;
    }
    
    // Reduce sum across threads
    float final_sum = block_reduce_sum(global_sum, reduce_smem);
    __syncthreads();
    
    // Reduce acc across threads and write output
    for (int d = 0; d < head_dim; d++) {
        float val = block_reduce_sum(acc[d], reduce_smem);
        __syncthreads();
        if (tid == 0) {
            Output[row * head_dim + d] = val / final_sum;
        }
    }
}

// Multi-head tiled version
__global__ void waller_operator_multihead_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Output,
    const int seq_len,
    const int head_dim,
    const int num_heads,
    const float scale
) {
    const int idx = blockIdx.x;
    const int head = idx / seq_len;
    const int row = idx % seq_len;
    if (head >= num_heads) return;
    
    const int tid = threadIdx.x;
    const int head_offset = head * seq_len * head_dim;
    
    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + TILE_SIZE * head_dim;
    float* reduce_smem = smem + 2 * TILE_SIZE * head_dim;
    
    float q_reg[128];
    for (int d = 0; d < head_dim; d++) {
        q_reg[d] = Q[head_offset + row * head_dim + d];
    }
    
    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float acc[128];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    for (int tile_start = 0; tile_start <= row; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, row + 1);
        int tile_len = tile_end - tile_start;
        
        for (int i = tid; i < tile_len * head_dim; i += BLOCK_SIZE) {
            int col_local = i / head_dim;
            int d = i % head_dim;
            int col_global = tile_start + col_local;
            K_tile[col_local * head_dim + d] = K[head_offset + col_global * head_dim + d];
            V_tile[col_local * head_dim + d] = V[head_offset + col_global * head_dim + d];
        }
        __syncthreads();
        
        for (int col_local = tid; col_local < tile_len; col_local += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_reg[d] * K_tile[col_local * head_dim + d];
            }
            dot *= scale;
            
            float old_max = global_max;
            global_max = fmaxf(global_max, dot);
            float rescale = expf(old_max - global_max);
            global_sum = global_sum * rescale + expf(dot - global_max);
            
            float weight = expf(dot - global_max);
            for (int d = 0; d < head_dim; d++) {
                acc[d] = acc[d] * rescale + weight * V_tile[col_local * head_dim + d];
            }
        }
        __syncthreads();
    }
    
    float final_max = block_reduce_max(global_max, reduce_smem);
    __syncthreads();
    float rescale = expf(global_max - final_max);
    global_sum *= rescale;
    for (int d = 0; d < head_dim; d++) acc[d] *= rescale;
    
    float final_sum = block_reduce_sum(global_sum, reduce_smem);
    __syncthreads();
    
    for (int d = 0; d < head_dim; d++) {
        float val = block_reduce_sum(acc[d], reduce_smem);
        __syncthreads();
        if (tid == 0) {
            Output[head_offset + row * head_dim + d] = val / final_sum;
        }
    }
}


void run_benchmark_v2(int seq_len, int head_dim, int num_iterations, int num_heads = 1) {
    printf("\n════════════════════════════════════════════════════════════════\n");
    
    size_t matrix_size = (size_t)seq_len * head_dim * num_heads * sizeof(float);
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t required = 4 * matrix_size + (1024 * 1024 * 100);
    
    if (required > free_mem) {
        printf("Seq: %d | Heads: %d | SKIPPED - Need %.2f GB, Have %.2f GB free\n", 
               seq_len, num_heads, required / 1e9, free_mem / 1e9);
        return;
    }
    
    printf("Seq: %d | Head: %d | Heads: %d | Iters: %d\n", seq_len, head_dim, num_heads, num_iterations);
    
    float *h_Q = (float*)malloc(matrix_size);
    float *h_K = (float*)malloc(matrix_size);
    float *h_V = (float*)malloc(matrix_size);
    
    srand(42);
    for (size_t i = 0; i < (size_t)seq_len * head_dim * num_heads; i++) {
        h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_K[i] = (float)rand() / RAND_MAX - 0.5f;
        h_V[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    float *d_Q, *d_K, *d_V, *d_Output;
    cudaMalloc(&d_Q, matrix_size);
    cudaMalloc(&d_K, matrix_size);
    cudaMalloc(&d_V, matrix_size);
    cudaMalloc(&d_Output, matrix_size);
    
    cudaMemcpy(d_Q, h_Q, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, matrix_size, cudaMemcpyHostToDevice);
    
    float scale = 1.0f / sqrtf((float)head_dim);
    int total_rows = seq_len * num_heads;
    size_t smem_size = (2 * TILE_SIZE * head_dim + 32) * sizeof(float);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        if (num_heads == 1) {
            waller_operator_tiled_kernel<<<seq_len, BLOCK_SIZE, smem_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, scale);
        } else {
            waller_operator_multihead_tiled_kernel<<<total_rows, BLOCK_SIZE, smem_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, num_heads, scale);
        }
    }
    cudaDeviceSynchronize();

    // Warmup
    for (int i = 0; i < 3; i++) {
        if (num_heads == 1) {
            waller_operator_tiled_kernel<<<seq_len, BLOCK_SIZE, smem_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, scale);
        } else {
            waller_operator_multihead_tiled_kernel<<<total_rows, BLOCK_SIZE, smem_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, num_heads, scale);
        }
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        if (num_heads == 1) {
            waller_operator_tiled_kernel<<<seq_len, BLOCK_SIZE, smem_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, scale);
        } else {
            waller_operator_multihead_tiled_kernel<<<total_rows, BLOCK_SIZE, smem_size>>>(
                d_Q, d_K, d_V, d_Output, seq_len, head_dim, num_heads, scale);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ms = milliseconds / num_iterations;
    
    long long flops = (long long)num_heads * seq_len * (seq_len + 1) / 2 * (4 * head_dim + 6);
    double tflops = (flops / 1e12) / (avg_time_ms / 1000.0);
    
    size_t standard_mem = (size_t)num_heads * seq_len * seq_len * sizeof(float) + 4 * matrix_size;
    size_t waller_mem = 4 * matrix_size;
    float mem_reduction = 100.0f * (1.0f - (float)waller_mem / (float)standard_mem);
    
    printf("Time: %.3f ms | %.2f TFLOPS | Mem Reduction: %.1f%%\n", avg_time_ms, tflops, mem_reduction);
    printf("Standard Attention Would Need: %.2f GB\n", standard_mem / 1e9);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Output);
    free(h_Q); free(h_K); free(h_V);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}


int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER OPERATOR V2 - TILED PARALLEL REDUCTION                ║\n");
    printf("║     Geodesic Attention Engine (GAE)                             ║\n");
    printf("║     Copyright 2026 Eric Waller                                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s | SM: %d.%d | SMs: %d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Memory: %.2f GB free / %.2f GB total\n", free_mem/1e9, total_mem/1e9);
    
    printf("\n[1] SINGLE HEAD SCALING TEST\n");
    run_benchmark_v2(4096, 128, 10, 1);
    run_benchmark_v2(16384, 128, 10, 1);
    run_benchmark_v2(65536, 128, 5, 1);
    run_benchmark_v2(131072, 128, 3, 1);
    
    printf("\n[2] MULTI-HEAD TRANSFORMER WORKLOAD\n");
    run_benchmark_v2(4096, 128, 10, 32);
    run_benchmark_v2(8192, 128, 10, 32);
    run_benchmark_v2(16384, 128, 5, 96);
    run_benchmark_v2(32768, 128, 3, 96);
    
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("BENCHMARK COMPLETE\n");
    printf("════════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}

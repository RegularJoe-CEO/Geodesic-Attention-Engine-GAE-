// Waller Operator INT8 - CUDA Implementation
// Geodesic Attention Engine (GAE)
// Copyright 2026 Eric Waller - Proprietary
// INT8 inputs with FP32 accumulation (Tesla-style mixed precision)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// INT8 dot product with INT32 accumulation (mimics tensor core behavior)
__device__ __forceinline__ int32_t dot_int8(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int head_dim
) {
    int32_t acc = 0;
    #pragma unroll 8
    for (int d = 0; d < head_dim; d++) {
        acc += (int32_t)a[d] * (int32_t)b[d];
    }
    return acc;
}

// Main INT8 Triangle Engine Kernel
__global__ void waller_operator_int8_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    float* __restrict__ Output,
    const int seq_len,
    const int head_dim,
    const float qk_scale,
    const float v_scale
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[128] = {0.0f};
    
    // Triangle: only attend to col <= row
    for (int col = 0; col <= row; col++) {
        int32_t dot_int = dot_int8(&Q[row * head_dim], &K[col * head_dim], head_dim);
        float dot = (float)dot_int * qk_scale;
        
        float old_max = max_val;
        max_val = fmaxf(max_val, dot);
        float rescale = expf(old_max - max_val);
        sum_exp = sum_exp * rescale + expf(dot - max_val);
        
        float weight = expf(dot - max_val);
        
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            float v_val = (float)V[col * head_dim + d] / v_scale;
            acc[d] = acc[d] * rescale + weight * v_val;
        }
    }
    
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < head_dim; d++) {
        Output[row * head_dim + d] = acc[d] * inv_sum;
    }
}

void run_int8_benchmark(int seq_len, int head_dim, int num_iterations) {
    printf("\n────────────────────────────────────────────────────────────────\n");
    printf("Seq: %d | Head: %d | Iters: %d\n", seq_len, head_dim, num_iterations);
    
    size_t int8_size = (size_t)seq_len * head_dim * sizeof(int8_t);
    size_t fp32_size = (size_t)seq_len * head_dim * sizeof(float);
    
    int8_t *h_Q = (int8_t*)malloc(int8_size);
    int8_t *h_K = (int8_t*)malloc(int8_size);
    int8_t *h_V = (int8_t*)malloc(int8_size);
    float *h_Output = (float*)malloc(fp32_size);
    
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
    
    float qk_scale = 1.0f / (sqrtf((float)head_dim) * 127.0f * 127.0f);
    float v_scale = 127.0f;
    
    int block_size = 256;
    int grid_size = (seq_len + block_size - 1) / block_size;
    
    for (int i = 0; i < 3; i++) {
        waller_operator_int8_kernel<<<grid_size, block_size>>>(
            d_Q, d_K, d_V, d_Output, seq_len, head_dim, qk_scale, v_scale
        );
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        waller_operator_int8_kernel<<<grid_size, block_size>>>(
            d_Q, d_K, d_V, d_Output, seq_len, head_dim, qk_scale, v_scale
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ms = milliseconds / num_iterations;
    
    size_t int8_mem = 3 * int8_size + fp32_size;
    size_t fp32_triangle_mem = 4 * fp32_size;
    size_t standard_attn = (size_t)seq_len * seq_len * sizeof(float) + fp32_triangle_mem;
    
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
    free(h_Q); free(h_K); free(h_V); free(h_Output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER OPERATOR INT8 - TRIANGLE ENGINE                      ║\n");
    printf("║     Tesla-Style Mixed Precision (INT8 compute, FP32 accum)      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    run_int8_benchmark(8192, 64, 20);
    run_int8_benchmark(16384, 64, 10);
    run_int8_benchmark(32768, 64, 5);
    run_int8_benchmark(65536, 64, 3);
    run_int8_benchmark(131072, 64, 2);
    run_int8_benchmark(262144, 64, 1);
    
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("INT8 BENCHMARK COMPLETE\n");
    
    return 0;
}

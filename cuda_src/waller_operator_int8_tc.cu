// Waller Operator INT8 - VECTORIZED VERSION
// Geodesic Attention Engine (GAE)
// Copyright 2026 Eric Waller - Proprietary
// Vectorized INT8 loads for better memory throughput

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Vectorized INT8 dot product using int4 (16 bytes = 16 INT8s at once)
__device__ __forceinline__ int32_t dot_int8_vectorized(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int head_dim
) {
    int32_t acc = 0;
    const int4* a_vec = reinterpret_cast<const int4*>(a);
    const int4* b_vec = reinterpret_cast<const int4*>(b);
    
    #pragma unroll
    for (int i = 0; i < head_dim / 16; i++) {
        int4 av = a_vec[i];
        int4 bv = b_vec[i];
        
        const int8_t* a8 = reinterpret_cast<const int8_t*>(&av);
        const int8_t* b8 = reinterpret_cast<const int8_t*>(&bv);
        
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            acc += (int32_t)a8[j] * (int32_t)b8[j];
        }
    }
    return acc;
}

// Vectorized INT8 Triangle Kernel
__global__ void waller_int8_vectorized_kernel(
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
    
    const int8_t* Q_row = &Q[row * head_dim];
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[128] = {0.0f};
    
    // Triangle: only attend to col <= row
    for (int col = 0; col <= row; col++) {
        int32_t dot = dot_int8_vectorized(Q_row, &K[col * head_dim], head_dim);
        float score = (float)dot * scale;
        
        float old_max = max_val;
        max_val = fmaxf(max_val, score);
        float rescale = expf(old_max - max_val);
        sum_exp = sum_exp * rescale + expf(score - max_val);
        
        float weight = expf(score - max_val);
        
        // Vectorized V load and accumulate
        const int4* V_vec = reinterpret_cast<const int4*>(&V[col * head_dim]);
        #pragma unroll
        for (int i = 0; i < head_dim / 16; i++) {
            int4 v4 = V_vec[i];
            const int8_t* v8 = reinterpret_cast<const int8_t*>(&v4);
            
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int idx = i * 16 + j;
                acc[idx] = acc[idx] * rescale + weight * (float)v8[j];
            }
        }
    }
    
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < head_dim; d++) {
        Output[row * head_dim + d] = acc[d] * inv_sum;
    }
}

void run_benchmark(int seq_len, int head_dim, int num_iterations) {
    printf("\n────────────────────────────────────────────────────────────────\n");
    printf("Seq: %d | Head: %d | Iters: %d\n", seq_len, head_dim, num_iterations);
    
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
    
    int block_size = 256;
    int grid_size = (seq_len + block_size - 1) / block_size;
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        waller_int8_vectorized_kernel<<<grid_size, block_size>>>(
            d_Q, d_K, d_V, d_Output, seq_len, head_dim, scale
        );
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        waller_int8_vectorized_kernel<<<grid_size, block_size>>>(
            d_Q, d_K, d_V, d_Output, seq_len, head_dim, scale
        );
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
    printf("║     WALLER INT8 VECTORIZED - TRIANGLE ENGINE                    ║\n");
    printf("║     16-byte Coalesced Loads (int4)                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s\n", prop.name);
    printf("Memory Bus: %d-bit | Bandwidth: %.0f GB/s\n", 
           prop.memoryBusWidth, 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    
    run_benchmark(8192, 64, 20);
    run_benchmark(16384, 64, 10);
    run_benchmark(32768, 64, 5);
    run_benchmark(65536, 64, 3);
    run_benchmark(131072, 64, 2);
    run_benchmark(262144, 64, 1);
    
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("VECTORIZED INT8 BENCHMARK COMPLETE\n");
    
    return 0;
}

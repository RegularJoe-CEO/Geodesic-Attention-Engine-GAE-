#include <cuda_runtime.h>
#include <cublas_v2.h>
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

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error at line %d\n", __LINE__); \
        exit(1); \
    } \
}

#define TILE_SIZE 256

// Fused softmax + V accumulation with online algorithm
__global__ void online_softmax_v_kernel(
    const float* __restrict__ scores,  // [tile_rows x tile_cols]
    const float* __restrict__ V_tile,  // [tile_cols x head_dim]
    float* __restrict__ O,             // [seq_len x head_dim]
    float* __restrict__ m_global,      // [seq_len] running max
    float* __restrict__ l_global,      // [seq_len] running sum
    int row_offset,
    int tile_cols,
    int seq_len,
    int head_dim,
    int tile_rows,
    float scale
) {
    int row_local = blockIdx.x;
    int row = row_offset + row_local;
    if (row >= seq_len || row_local >= tile_rows) return;
    
    int tid = threadIdx.x;
    
    // Load current max and sum
    float m_old = m_global[row];
    float l_old = l_global[row];
    
    // Find max in this tile
    float m_tile = -INFINITY;
    for (int c = 0; c < tile_cols; c++) {
        float s = scores[row_local * tile_cols + c] * scale;
        m_tile = fmaxf(m_tile, s);
    }
    
    // New global max
    float m_new = fmaxf(m_old, m_tile);
    
    // Compute sum of exp for this tile and rescale factor
    float exp_sum = 0.0f;
    for (int c = 0; c < tile_cols; c++) {
        float s = scores[row_local * tile_cols + c] * scale;
        exp_sum += expf(s - m_new);
    }
    
    float rescale = expf(m_old - m_new);
    float l_new = l_old * rescale + exp_sum;
    
    // Update output: rescale old accumulator and add new
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float old_val = O[row * head_dim + d];
        float new_val = 0.0f;
        
        for (int c = 0; c < tile_cols; c++) {
            float s = scores[row_local * tile_cols + c] * scale;
            float weight = expf(s - m_new);
            new_val += weight * V_tile[c * head_dim + d];
        }
        
        O[row * head_dim + d] = old_val * rescale + new_val;
    }
    
    // Update global max and sum
    if (tid == 0) {
        m_global[row] = m_new;
        l_global[row] = l_new;
    }
}

// Final normalization
__global__ void normalize_output(
    float* __restrict__ O,
    const float* __restrict__ l_global,
    int seq_len,
    int head_dim
) {
    int row = blockIdx.x;
    if (row >= seq_len) return;
    
    float l = l_global[row];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        O[row * head_dim + d] /= l;
    }
}

void run_tiled_attention(
    cublasHandle_t handle,
    float* d_Q, float* d_K, float* d_V, float* d_O,
    float* d_scores, float* d_m, float* d_l,
    int seq_len, int head_dim, float scale
) {
    // Initialize m to -inf, l to 0, O to 0
    CHECK_CUDA(cudaMemset(d_O, 0, seq_len * head_dim * sizeof(float)));
    
    float* h_m = (float*)malloc(seq_len * sizeof(float));
    float* h_l = (float*)malloc(seq_len * sizeof(float));
    for (int i = 0; i < seq_len; i++) {
        h_m[i] = -INFINITY;
        h_l[i] = 0.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_m, h_m, seq_len * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_l, h_l, seq_len * sizeof(float), cudaMemcpyHostToDevice));
    free(h_m); free(h_l);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Process in tiles
    for (int row_start = 0; row_start < seq_len; row_start += TILE_SIZE) {
        int tile_rows = min(TILE_SIZE, seq_len - row_start);
        
        // For causal, max column is row_start + tile_rows
        int max_col = min(row_start + tile_rows, seq_len);
        
        for (int col_start = 0; col_start < max_col; col_start += TILE_SIZE) {
            int tile_cols = min(TILE_SIZE, max_col - col_start);
            
            // Compute Q_tile @ K_tile^T using cuBLAS
            // Q_tile: [tile_rows x head_dim], K_tile: [tile_cols x head_dim]
            // scores: [tile_rows x tile_cols]
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                tile_cols, tile_rows, head_dim,
                &alpha,
                d_K + col_start * head_dim, head_dim,
                d_Q + row_start * head_dim, head_dim,
                &beta,
                d_scores, tile_cols));
            
            // Online softmax + V accumulation
            online_softmax_v_kernel<<<tile_rows, 128>>>(
                d_scores,
                d_V + col_start * head_dim,
                d_O, d_m, d_l,
                row_start, tile_cols, seq_len, head_dim, tile_rows, scale
            );
        }
    }
    
    // Final normalization
    normalize_output<<<seq_len, 128>>>(d_O, d_l, seq_len, head_dim);
}

void run_benchmark(int seq_len, int head_dim, int iters) {
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("Seq: %d | HeadDim: %d | Iters: %d\n", seq_len, head_dim, iters);
    
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    size_t mat_size = (size_t)seq_len * head_dim * sizeof(float);
    size_t score_size = (size_t)TILE_SIZE * TILE_SIZE * sizeof(float);
    
    float *h_Q = (float*)malloc(mat_size);
    srand(42);
    for (size_t i = 0; i < (size_t)seq_len * head_dim; i++) {
        h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    float *d_Q, *d_K, *d_V, *d_O, *d_scores, *d_m, *d_l;
    CHECK_CUDA(cudaMalloc(&d_Q, mat_size));
    CHECK_CUDA(cudaMalloc(&d_K, mat_size));
    CHECK_CUDA(cudaMalloc(&d_V, mat_size));
    CHECK_CUDA(cudaMalloc(&d_O, mat_size));
    CHECK_CUDA(cudaMalloc(&d_scores, score_size));
    CHECK_CUDA(cudaMalloc(&d_m, seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l, seq_len * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_Q, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_Q, mat_size, cudaMemcpyHostToDevice));
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Warmup
    for (int i = 0; i < 2; i++) {
        run_tiled_attention(handle, d_Q, d_K, d_V, d_O, d_scores, d_m, d_l, seq_len, head_dim, scale);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        run_tiled_attention(handle, d_Q, d_K, d_V, d_O, d_scores, d_m, d_l, seq_len, head_dim, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    
    // FLOPS: 2*N*N*d for QK^T, 2*N*N*d for softmax@V, plus softmax ops
    long long flops = 4LL * seq_len * seq_len * head_dim;
    double tflops = (flops / 1e12) / (avg_ms / 1000.0);
    
    size_t standard_mem = (size_t)seq_len * seq_len * sizeof(float);
    size_t waller_mem = 4 * mat_size + score_size + 2 * seq_len * sizeof(float);
    float mem_reduction = 100.0f * (1.0f - (float)waller_mem / (standard_mem + 4 * mat_size));
    
    printf("Time: %.3f ms | %.2f TFLOPS | Mem Reduction: %.1f%%\n", avg_ms, tflops, mem_reduction);
    printf("Standard NxN: %.2f GB | Waller: %.4f GB (tile: %d)\n", 
           standard_mem / 1e9, waller_mem / 1e9, TILE_SIZE);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_scores); cudaFree(d_m); cudaFree(d_l);
    free(h_Q);
    cublasDestroy(handle);
}

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER OPERATOR V6 - cuBLAS TILED                            ║\n");
    printf("║     Copyright 2026 Eric Waller - Patent Pending                  ║\n");
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

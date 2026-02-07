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

#define TILE_SIZE 512

__global__ void fused_softmax_v_update(
    const float* __restrict__ scores,
    const float* __restrict__ V_tile,
    float* __restrict__ O,
    float* __restrict__ m_global,
    float* __restrict__ l_global,
    int row_offset,
    int col_offset,
    int tile_rows,
    int tile_cols,
    int score_ld,
    int seq_len,
    int head_dim,
    float scale
) {
    int local_row = blockIdx.x;
    int row = row_offset + local_row;
    if (row >= seq_len || local_row >= tile_rows) return;
    
    int tid = threadIdx.x;
    
    int max_col_for_row = min(tile_cols, row - col_offset + 1);
    if (max_col_for_row <= 0) return;
    
    float m_old = m_global[row];
    float l_old = l_global[row];
    
    float m_tile = -INFINITY;
    for (int c = 0; c < max_col_for_row; c++) {
        float s = scores[local_row * score_ld + c] * scale;
        m_tile = fmaxf(m_tile, s);
    }
    
    float m_new = fmaxf(m_old, m_tile);
    float rescale = expf(m_old - m_new);
    
    float exp_sum = 0.0f;
    for (int c = 0; c < max_col_for_row; c++) {
        float s = scores[local_row * score_ld + c] * scale;
        exp_sum += expf(s - m_new);
    }
    
    float l_new = l_old * rescale + exp_sum;
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = O[row * head_dim + d] * rescale;
        for (int c = 0; c < max_col_for_row; c++) {
            float s = scores[local_row * score_ld + c] * scale;
            float w = expf(s - m_new);
            acc += w * V_tile[c * head_dim + d];
        }
        O[row * head_dim + d] = acc;
    }
    
    if (tid == 0) {
        m_global[row] = m_new;
        l_global[row] = l_new;
    }
}

__global__ void normalize_output(float* O, const float* l, int seq_len, int head_dim) {
    int row = blockIdx.x;
    if (row >= seq_len) return;
    float norm = l[row];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        O[row * head_dim + d] /= norm;
    }
}

__global__ void init_ml(float* m, float* l, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
}

void run_benchmark(int seq_len, int head_dim, int iters) {
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("Seq: %d | HeadDim: %d | Iters: %d\n", seq_len, head_dim, iters);
    
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    size_t mat_size = (size_t)seq_len * head_dim * sizeof(float);
    size_t score_tile_size = (size_t)TILE_SIZE * TILE_SIZE * sizeof(float);
    
    float *h_data = (float*)malloc(mat_size);
    srand(42);
    for (size_t i = 0; i < (size_t)seq_len * head_dim; i++) {
        h_data[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    float *d_Q, *d_K, *d_V, *d_O, *d_scores, *d_m, *d_l;
    CHECK_CUDA(cudaMalloc(&d_Q, mat_size));
    CHECK_CUDA(cudaMalloc(&d_K, mat_size));
    CHECK_CUDA(cudaMalloc(&d_V, mat_size));
    CHECK_CUDA(cudaMalloc(&d_O, mat_size));
    CHECK_CUDA(cudaMalloc(&d_scores, score_tile_size));
    CHECK_CUDA(cudaMalloc(&d_m, seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l, seq_len * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_Q, h_data, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_data, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_data, mat_size, cudaMemcpyHostToDevice));
    
    float scale = 1.0f / sqrtf((float)head_dim);
    float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    CHECK_CUDA(cudaMemset(d_O, 0, mat_size));
    init_ml<<<(seq_len+255)/256, 256>>>(d_m, d_l, seq_len);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < iters; iter++) {
        CHECK_CUDA(cudaMemset(d_O, 0, mat_size));
        init_ml<<<(seq_len+255)/256, 256>>>(d_m, d_l, seq_len);
        
        for (int row_start = 0; row_start < seq_len; row_start += TILE_SIZE) {
            int tile_rows = min(TILE_SIZE, seq_len - row_start);
            int max_col = row_start + tile_rows;
            
            for (int col_start = 0; col_start < max_col; col_start += TILE_SIZE) {
                int tile_cols = min(TILE_SIZE, seq_len - col_start);
                
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    tile_cols, tile_rows, head_dim,
                    &alpha,
                    d_K + col_start * head_dim, head_dim,
                    d_Q + row_start * head_dim, head_dim,
                    &beta,
                    d_scores, TILE_SIZE));
                
                fused_softmax_v_update<<<tile_rows, 128>>>(
                    d_scores, d_V + col_start * head_dim,
                    d_O, d_m, d_l,
                    row_start, col_start, tile_rows, tile_cols, TILE_SIZE,
                    seq_len, head_dim, scale);
            }
        }
        
        normalize_output<<<seq_len, 128>>>(d_O, d_l, seq_len, head_dim);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    
    long long flops = 4LL * seq_len * seq_len * head_dim;
    double tflops = (flops / 1e12) / (avg_ms / 1000.0);
    
    size_t standard_mem = (size_t)seq_len * seq_len * sizeof(float);
    size_t waller_mem = 4 * mat_size + score_tile_size + 2 * seq_len * sizeof(float);
    float mem_reduction = 100.0f * (1.0f - (float)waller_mem / (standard_mem + 4 * mat_size));
    
    printf("Time: %.2f ms | %.2f TFLOPS | Mem Red: %.1f%%\n", avg_ms, tflops, mem_reduction);
    printf("Standard: %.2f GB | Waller: %.4f GB\n", standard_mem / 1e9, waller_mem / 1e9);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_scores); cudaFree(d_m); cudaFree(d_l);
    free(h_data);
    cublasDestroy(handle);
}

int main() {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER OPERATOR V7 - TILED cuBLAS                            ║\n");
    printf("║     Copyright 2026 Eric Waller                                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s | SMs: %d\n", prop.name, prop.multiProcessorCount);
    
    run_benchmark(2048, 128, 5);
    run_benchmark(4096, 128, 5);
    run_benchmark(8192, 128, 3);
    run_benchmark(16384, 128, 2);
    run_benchmark(32768, 128, 1);
    run_benchmark(65536, 128, 1);
    
    printf("\nBENCHMARK COMPLETE\n");
    return 0;
}

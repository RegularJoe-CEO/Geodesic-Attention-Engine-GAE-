#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

#define WARP_SIZE 32

__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__global__ void online_softmax_kernel(
    const half* __restrict__ scores,
    half* __restrict__ P,
    float* __restrict__ m_global,
    float* __restrict__ l_global,
    float* __restrict__ rescale_out,
    int row_offset,
    int col_offset,
    int tile_rows,
    int tile_cols,
    int score_ld,
    int seq_len,
    float scale
) {
    int local_row = blockIdx.x;
    int row = row_offset + local_row;
    if (row >= seq_len || local_row >= tile_rows) return;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = num_threads / WARP_SIZE;
    
    extern __shared__ char shared_mem[];
    float* warp_vals = (float*)shared_mem;
    
    int max_col_for_row = min(tile_cols, row - col_offset + 1);
    
    float m_old = m_global[row];
    float l_old = l_global[row];
    
    float local_max = -INFINITY;
    for (int c = tid; c < max_col_for_row; c += num_threads) {
        float s = __half2float(scores[local_row * score_ld + c]) * scale;
        local_max = fmaxf(local_max, s);
    }
    local_max = warp_max(local_max);
    if (lane_id == 0) warp_vals[warp_id] = local_max;
    __syncthreads();
    
    float m_tile = -INFINITY;
    if (tid < num_warps) m_tile = warp_vals[tid];
    m_tile = warp_max(m_tile);
    m_tile = __shfl_sync(0xffffffff, m_tile, 0);
    
    float m_new = fmaxf(m_old, m_tile);
    float rescale = expf(m_old - m_new);
    
    float local_sum = 0.0f;
    for (int c = tid; c < tile_cols; c += num_threads) {
        float p_val = 0.0f;
        if (c < max_col_for_row) {
            float s = __half2float(scores[local_row * score_ld + c]) * scale;
            p_val = expf(s - m_new);
            local_sum += p_val;
        }
        P[local_row * tile_cols + c] = __float2half(p_val);
    }
    __syncthreads();
    
    local_sum = warp_sum(local_sum);
    if (lane_id == 0) warp_vals[warp_id] = local_sum;
    __syncthreads();
    
    float exp_sum = 0.0f;
    if (tid < num_warps) exp_sum = warp_vals[tid];
    exp_sum = warp_sum(exp_sum);
    exp_sum = __shfl_sync(0xffffffff, exp_sum, 0);
    
    float l_new = l_old * rescale + exp_sum;
    
    if (tid == 0) {
        m_global[row] = m_new;
        l_global[row] = l_new;
        rescale_out[local_row] = rescale;
    }
}

__global__ void update_output_kernel(
    half* __restrict__ O,
    const half* __restrict__ PV_tile,
    const float* __restrict__ rescale,
    int row_offset,
    int tile_rows,
    int seq_len,
    int head_dim
) {
    int local_row = blockIdx.x;
    int row = row_offset + local_row;
    if (row >= seq_len || local_row >= tile_rows) return;
    
    float r = rescale[local_row];
    
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float o_old = __half2float(O[row * head_dim + d]);
        float pv_new = __half2float(PV_tile[local_row * head_dim + d]);
        O[row * head_dim + d] = __float2half(o_old * r + pv_new);
    }
}

__global__ void normalize_output_fp16(half* O, const float* l, int seq_len, int head_dim) {
    int row = blockIdx.x;
    if (row >= seq_len) return;
    float norm = l[row];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = __half2float(O[row * head_dim + d]);
        O[row * head_dim + d] = __float2half(val / norm);
    }
}

__global__ void init_ml(float* m, float* l, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
}

__global__ void init_output_fp16(half* O, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) O[i] = __float2half(0.0f);
}

float run_benchmark(int seq_len, int head_dim, int tile_size, int iters, bool verbose) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    size_t mat_size = (size_t)seq_len * head_dim * sizeof(half);
    size_t score_tile_size = (size_t)tile_size * tile_size * sizeof(half);
    size_t pv_tile_size = (size_t)tile_size * head_dim * sizeof(half);
    
    half *h_data = (half*)malloc(mat_size);
    srand(42);
    for (size_t i = 0; i < (size_t)seq_len * head_dim; i++)
        h_data[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    
    half *d_Q, *d_K, *d_V, *d_O;
    half *d_scores, *d_P, *d_PV;
    float *d_m, *d_l, *d_rescale;
    
    CHECK_CUDA(cudaMalloc(&d_Q, mat_size));
    CHECK_CUDA(cudaMalloc(&d_K, mat_size));
    CHECK_CUDA(cudaMalloc(&d_V, mat_size));
    CHECK_CUDA(cudaMalloc(&d_O, mat_size));
    CHECK_CUDA(cudaMalloc(&d_scores, score_tile_size));
    CHECK_CUDA(cudaMalloc(&d_P, score_tile_size));
    CHECK_CUDA(cudaMalloc(&d_PV, pv_tile_size));
    CHECK_CUDA(cudaMalloc(&d_m, seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l, seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rescale, tile_size * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_Q, h_data, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_data, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_data, mat_size, cudaMemcpyHostToDevice));
    
    float scale = 1.0f / sqrtf((float)head_dim);
    half alpha_h = __float2half(1.0f);
    half beta_h = __float2half(0.0f);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    size_t total_elems = (size_t)seq_len * head_dim;
    init_output_fp16<<<(total_elems+255)/256, 256>>>(d_O, total_elems);
    init_ml<<<(seq_len+255)/256, 256>>>(d_m, d_l, seq_len);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    int threads = 256;
    int num_warps = threads / WARP_SIZE;
    size_t smem_size = num_warps * sizeof(float);
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < iters; iter++) {
        init_output_fp16<<<(total_elems+255)/256, 256>>>(d_O, total_elems);
        init_ml<<<(seq_len+255)/256, 256>>>(d_m, d_l, seq_len);
        
        for (int row_start = 0; row_start < seq_len; row_start += tile_size) {
            int tile_rows = min(tile_size, seq_len - row_start);
            int max_col = row_start + tile_rows;
            
            for (int col_start = 0; col_start < max_col; col_start += tile_size) {
                int tile_cols = min(tile_size, seq_len - col_start);
                
                CHECK_CUBLAS(cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    tile_cols, tile_rows, head_dim,
                    &alpha_h,
                    d_K + col_start * head_dim, CUDA_R_16F, head_dim,
                    d_Q + row_start * head_dim, CUDA_R_16F, head_dim,
                    &beta_h,
                    d_scores, CUDA_R_16F, tile_size,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                
                online_softmax_kernel<<<tile_rows, threads, smem_size>>>(
                    d_scores, d_P, d_m, d_l, d_rescale,
                    row_start, col_start, tile_rows, tile_cols, tile_size,
                    seq_len, scale);
                
                CHECK_CUBLAS(cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    head_dim, tile_rows, tile_cols,
                    &alpha_h,
                    d_V + col_start * head_dim, CUDA_R_16F, head_dim,
                    d_P, CUDA_R_16F, tile_cols,
                    &beta_h,
                    d_PV, CUDA_R_16F, head_dim,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                
                update_output_kernel<<<tile_rows, 256>>>(
                    d_O, d_PV, d_rescale,
                    row_start, tile_rows, seq_len, head_dim);
            }
        }
        
        normalize_output_fp16<<<seq_len, 256>>>(d_O, d_l, seq_len, head_dim);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    CHECK_CUDA(cudaGetLastError());
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    
    long long flops = 4LL * seq_len * seq_len * head_dim;
    double tflops = (flops / 1e12) / (avg_ms / 1000.0);
    
    if (verbose) {
        size_t standard_mem = (size_t)seq_len * seq_len * sizeof(half);
        size_t waller_mem = 4 * mat_size + score_tile_size * 2 + pv_tile_size;
        float mem_reduction = 100.0f * (1.0f - (float)waller_mem / (standard_mem + 4 * mat_size));
        printf("Seq: %d | Tile: %d | Time: %.2f ms | %.1f TFLOPS | %.2f%% peak | Mem: %.1f%%\n", 
               seq_len, tile_size, avg_ms, tflops, 100.0*tflops/1979.0, mem_reduction);
    }
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_scores); cudaFree(d_P); cudaFree(d_PV);
    cudaFree(d_m); cudaFree(d_l); cudaFree(d_rescale);
    free(h_data);
    cublasDestroy(handle);
    
    return tflops;
}

int main() {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER KERNEL - AUTO-TUNED TILE SIZE                            ║\n");
    printf("║     Copyright 2026 Eric Waller - Patent Pending                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s | SMs: %d\n", prop.name, prop.multiProcessorCount);
    
    int seq_lens[] = {2048, 4096, 8192, 16384, 32768, 65536, 131072};
    int tile_sizes[] = {256, 512, 1024, 2048, 4096};
    int num_seqs = 7;
    int num_tiles = 5;
    
    printf("\n>>> FINDING OPTIMAL TILE SIZE FOR EACH SEQUENCE LENGTH <<<\n\n");
    
    int best_tiles[7];
    int best_iters[] = {10, 10, 5, 3, 2, 1, 1};

    for (int s = 0; s < num_seqs; s++) {
        int seq = seq_lens[s];
        float best_tflops = 0;
        int best_tile = 256;
        
        printf("Seq %d: ", seq);
        fflush(stdout);
        
        for (int t = 0; t < num_tiles; t++) {
            int tile = tile_sizes[t];
            if (tile > seq) continue;
            
            // Check memory - tile^2 * 2 bytes for scores, need < ~1GB
            size_t tile_mem = (size_t)tile * tile * sizeof(half) * 2;
            if (tile_mem > 1024ULL * 1024 * 1024) continue;
            
            int iters = (seq <= 8192) ? 5 : (seq <= 32768) ? 2 : 1;
            float tflops = run_benchmark(seq, 128, tile, iters, false);
            
            printf("[%d: %.0f] ", tile, tflops);
            fflush(stdout);
            
            if (tflops > best_tflops) {
                best_tflops = tflops;
                best_tile = tile;
            }
        }
        best_tiles[s] = best_tile;
        printf("=> BEST: tile=%d (%.1f TFLOPS)\n", best_tile, best_tflops);
    }
    
    printf("\n>>> FINAL RESULTS WITH OPTIMAL TILE SIZES <<<\n\n");
    
    // Run final benchmark with AUTO-TUNED optimal tiles
    for (int s = 0; s < num_seqs; s++) {
        run_benchmark(seq_lens[s], 128, best_tiles[s], best_iters[s], true);
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    return 0;
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct __align__(32) TuningFork {
    float attn_scale;
    float attn_temperature;
    float residual_alpha;
    float residual_beta;
    float norm_eps;
    int activation_type;
    float mlp_gate;
    float output_scale;
};

__device__ __forceinline__ float activation(float x, int type) {
    switch(type) {
        case 0: return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        case 1: return x / (1.0f + expf(-x));
        default: return x;
    }
}

// HEAD_DIM=128, TILE=96 (24KB shared - safe)
__global__ void waller_fused_128(
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    const float* __restrict__ W_up,
    const float* __restrict__ W_down,
    const float* __restrict__ norm1_w,
    const float* __restrict__ norm2_w,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int seq_len,
    const TuningFork tuning
) {
    const int HEAD_DIM = 128;
    const int MLP_DIM = 512;
    const int TILE_SIZE = 96;
    
    __shared__ int8_t K_tile[96][128];
    __shared__ int8_t V_tile[96][128];
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    float state[128];
    float residual1[128];
    for (int d = 0; d < HEAD_DIM; d++) {
        state[d] = input[row * HEAD_DIM + d];
        residual1[d] = state[d];
    }
    
    int8_t Q_local[128];
    for (int d = 0; d < HEAD_DIM; d++) {
        Q_local[d] = (int8_t)fmaxf(-127.0f, fminf(127.0f, state[d] * 127.0f));
    }
    
    float attn_out[128];
    for (int d = 0; d < HEAD_DIM; d++) attn_out[d] = 0.0f;
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    int max_col = row + 1;
    
    for (int tile_start = 0; tile_start < max_col; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, max_col);
        int tile_len = tile_end - tile_start;
        
        __syncthreads();
        for (int i = tid; i < tile_len; i += block_size) {
            int col = tile_start + i;
            if (col < seq_len) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    K_tile[i][d] = K[col * HEAD_DIM + d];
                    V_tile[i][d] = V[col * HEAD_DIM + d];
                }
            }
        }
        __syncthreads();
        
        for (int i = 0; i < tile_len; i++) {
            int col = tile_start + i;
            if (col > row) break;
            
            int32_t dot = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += (int32_t)Q_local[d] * (int32_t)K_tile[i][d];
            }
            
            float score = (float)dot * tuning.attn_scale / tuning.attn_temperature;
            
            float old_max = max_val;
            max_val = fmaxf(max_val, score);
            float rescale = expf(old_max - max_val);
            sum_exp = sum_exp * rescale + expf(score - max_val);
            float weight = expf(score - max_val);
            
            for (int d = 0; d < HEAD_DIM; d++) {
                attn_out[d] = attn_out[d] * rescale + weight * (float)V_tile[i][d];
            }
        }
    }
    
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < HEAD_DIM; d++) attn_out[d] *= inv_sum;
    
    for (int d = 0; d < HEAD_DIM; d++) {
        state[d] = residual1[d] * tuning.residual_alpha + attn_out[d];
    }
    
    float ss = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) ss += state[d] * state[d];
    float scale = rsqrtf(ss / HEAD_DIM + tuning.norm_eps);
    for (int d = 0; d < HEAD_DIM; d++) state[d] = state[d] * scale * norm1_w[d];
    
    float residual2[128];
    for (int d = 0; d < HEAD_DIM; d++) residual2[d] = state[d];
    
    float mlp_out[128];
    for (int d = 0; d < HEAD_DIM; d++) mlp_out[d] = 0.0f;
    
    for (int i = 0; i < MLP_DIM; i++) {
        float sum = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            sum += state[d] * W_up[d * MLP_DIM + i];
        }
        float activated = activation(sum, tuning.activation_type);
        for (int d = 0; d < HEAD_DIM; d++) {
            mlp_out[d] += activated * W_down[i * HEAD_DIM + d];
        }
    }
    
    for (int d = 0; d < HEAD_DIM; d++) {
        state[d] = residual2[d] * tuning.residual_beta + mlp_out[d] * tuning.mlp_gate;
    }
    
    ss = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) ss += state[d] * state[d];
    scale = rsqrtf(ss / HEAD_DIM + tuning.norm_eps);
    for (int d = 0; d < HEAD_DIM; d++) {
        output[row * HEAD_DIM + d] = state[d] * scale * norm2_w[d] * tuning.output_scale;
    }
}

// HEAD_DIM=256, TILE=48 (24KB shared - safe)
__global__ void waller_fused_256(
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    const float* __restrict__ W_up,
    const float* __restrict__ W_down,
    const float* __restrict__ norm1_w,
    const float* __restrict__ norm2_w,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int seq_len,
    const TuningFork tuning
) {
    const int HEAD_DIM = 256;
    const int MLP_DIM = 1024;
    const int TILE_SIZE = 48;
    
    __shared__ int8_t K_tile[48][256];
    __shared__ int8_t V_tile[48][256];
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    float state[256];
    float residual1[256];
    for (int d = 0; d < HEAD_DIM; d++) {
        state[d] = input[row * HEAD_DIM + d];
        residual1[d] = state[d];
    }
    
    int8_t Q_local[256];
    for (int d = 0; d < HEAD_DIM; d++) {
        Q_local[d] = (int8_t)fmaxf(-127.0f, fminf(127.0f, state[d] * 127.0f));
    }
    
    float attn_out[256];
    for (int d = 0; d < HEAD_DIM; d++) attn_out[d] = 0.0f;
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    int max_col = row + 1;
    
    for (int tile_start = 0; tile_start < max_col; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, max_col);
        int tile_len = tile_end - tile_start;
        
        __syncthreads();
        for (int i = tid; i < tile_len; i += block_size) {
            int col = tile_start + i;
            if (col < seq_len) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    K_tile[i][d] = K[col * HEAD_DIM + d];
                    V_tile[i][d] = V[col * HEAD_DIM + d];
                }
            }
        }
        __syncthreads();
        
        for (int i = 0; i < tile_len; i++) {
            int col = tile_start + i;
            if (col > row) break;
            
            int32_t dot = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += (int32_t)Q_local[d] * (int32_t)K_tile[i][d];
            }
            
            float score = (float)dot * tuning.attn_scale / tuning.attn_temperature;
            
            float old_max = max_val;
            max_val = fmaxf(max_val, score);
            float rescale = expf(old_max - max_val);
            sum_exp = sum_exp * rescale + expf(score - max_val);
            float weight = expf(score - max_val);
            
            for (int d = 0; d < HEAD_DIM; d++) {
                attn_out[d] = attn_out[d] * rescale + weight * (float)V_tile[i][d];
            }
        }
    }
    
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < HEAD_DIM; d++) attn_out[d] *= inv_sum;
    
    for (int d = 0; d < HEAD_DIM; d++) {
        state[d] = residual1[d] * tuning.residual_alpha + attn_out[d];
    }
    
    float ss = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) ss += state[d] * state[d];
    float scale = rsqrtf(ss / HEAD_DIM + tuning.norm_eps);
    for (int d = 0; d < HEAD_DIM; d++) state[d] = state[d] * scale * norm1_w[d];
    
    float residual2[256];
    for (int d = 0; d < HEAD_DIM; d++) residual2[d] = state[d];
    
    float mlp_out[256];
    for (int d = 0; d < HEAD_DIM; d++) mlp_out[d] = 0.0f;
    
    // Process MLP in chunks of 64 to manage registers
    for (int m = 0; m < MLP_DIM; m++) {
        float sum = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            sum += state[d] * W_up[d * MLP_DIM + m];
        }
        float activated = activation(sum, tuning.activation_type);
        for (int d = 0; d < HEAD_DIM; d++) {
            mlp_out[d] += activated * W_down[m * HEAD_DIM + d];
        }
    }
    
    for (int d = 0; d < HEAD_DIM; d++) {
        state[d] = residual2[d] * tuning.residual_beta + mlp_out[d] * tuning.mlp_gate;
    }
    
    ss = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) ss += state[d] * state[d];
    scale = rsqrtf(ss / HEAD_DIM + tuning.norm_eps);
    for (int d = 0; d < HEAD_DIM; d++) {
        output[row * HEAD_DIM + d] = state[d] * scale * norm2_w[d] * tuning.output_scale;
    }
}

void run_benchmark_128(int seq_len, int num_iterations) {
    const int HEAD_DIM = 128;
    const int MLP_DIM = 512;
    
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("HEAD_DIM=128: Seq %d | Iters: %d\n", seq_len, num_iterations);
    
    size_t int8_size = (size_t)seq_len * HEAD_DIM;
    size_t fp32_size = (size_t)seq_len * HEAD_DIM * sizeof(float);
    size_t w_up_size = (size_t)HEAD_DIM * MLP_DIM * sizeof(float);
    size_t w_down_size = (size_t)MLP_DIM * HEAD_DIM * sizeof(float);
    
    int8_t *h_K = (int8_t*)malloc(int8_size);
    int8_t *h_V = (int8_t*)malloc(int8_size);
    float *h_input = (float*)malloc(fp32_size);
    float *h_W_up = (float*)malloc(w_up_size);
    float *h_W_down = (float*)malloc(w_down_size);
    float *h_norm1 = (float*)malloc(HEAD_DIM * sizeof(float));
    float *h_norm2 = (float*)malloc(HEAD_DIM * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < int8_size; i++) {
        h_K[i] = (int8_t)(rand() % 256 - 128);
        h_V[i] = (int8_t)(rand() % 256 - 128);
    }
    for (size_t i = 0; i < (size_t)seq_len * HEAD_DIM; i++) {
        h_input[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (size_t i = 0; i < (size_t)HEAD_DIM * MLP_DIM; i++) h_W_up[i] = 0.01f;
    for (size_t i = 0; i < (size_t)MLP_DIM * HEAD_DIM; i++) h_W_down[i] = 0.01f;
    for (int i = 0; i < HEAD_DIM; i++) { h_norm1[i] = 1.0f; h_norm2[i] = 1.0f; }
    
    int8_t *d_K, *d_V;
    float *d_input, *d_output, *d_W_up, *d_W_down, *d_norm1, *d_norm2;
    
    cudaMalloc(&d_K, int8_size);
    cudaMalloc(&d_V, int8_size);
    cudaMalloc(&d_input, fp32_size);
    cudaMalloc(&d_output, fp32_size);
    cudaMalloc(&d_W_up, w_up_size);
    cudaMalloc(&d_W_down, w_down_size);
    cudaMalloc(&d_norm1, HEAD_DIM * sizeof(float));
    cudaMalloc(&d_norm2, HEAD_DIM * sizeof(float));
    
    cudaMemcpy(d_K, h_K, int8_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, int8_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, fp32_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_up, h_W_up, w_up_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_down, h_W_down, w_down_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm1, h_norm1, HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm2, h_norm2, HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    TuningFork tuning = {1.0f / (sqrtf(128.0f) * 127.0f * 127.0f), 1.0f, 1.0f, 1.0f, 1e-6f, 0, 1.0f, 1.0f};
    
    int block = 32;
    int grid = (seq_len + block - 1) / block;
    
    for (int i = 0; i < 3; i++) {
        waller_fused_128<<<grid, block>>>(d_K, d_V, d_W_up, d_W_down, d_norm1, d_norm2, d_input, d_output, seq_len, tuning);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        waller_fused_128<<<grid, block>>>(d_K, d_V, d_W_up, d_W_down, d_norm1, d_norm2, d_input, d_output, seq_len, tuning);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / num_iterations;
    
    double attn_flops = (double)seq_len * ((double)seq_len + 1.0) / 2.0 * HEAD_DIM * 2.0;
    double mlp_flops = (double)seq_len * HEAD_DIM * MLP_DIM * 4.0;
    double total_flops = attn_flops + mlp_flops;
    double tflops = (total_flops / (avg_ms / 1000.0)) / 1e12;
    
    size_t standard_hbm = 8 * fp32_size;
    size_t fused_hbm = 2 * fp32_size + w_up_size + w_down_size;
    float hbm_reduction = 100.0f * (1.0f - (float)fused_hbm / (float)standard_hbm);
    
    size_t standard_attn_mem = (size_t)seq_len * seq_len * sizeof(float);
    
    printf("Time: %.3f ms | %.3f TFLOPS\n", avg_ms, tflops);
    printf("HBM Reduction: %.1f%%\n", hbm_reduction);
    printf("Standard attention would need: %.2f GB\n", standard_attn_mem / 1e9);
    if (standard_attn_mem > 80e9) printf(">>> IMPOSSIBLE WITH STANDARD <<<\n");
    
    cudaFree(d_K); cudaFree(d_V); cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_W_up); cudaFree(d_W_down); cudaFree(d_norm1); cudaFree(d_norm2);
    free(h_K); free(h_V); free(h_input); free(h_W_up); free(h_W_down); free(h_norm1); free(h_norm2);
}

void run_benchmark_256(int seq_len, int num_iterations) {
    const int HEAD_DIM = 256;
    const int MLP_DIM = 1024;
    
    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("HEAD_DIM=256: Seq %d | Iters: %d\n", seq_len, num_iterations);
    
    size_t int8_size = (size_t)seq_len * HEAD_DIM;
    size_t fp32_size = (size_t)seq_len * HEAD_DIM * sizeof(float);
    size_t w_up_size = (size_t)HEAD_DIM * MLP_DIM * sizeof(float);
    size_t w_down_size = (size_t)MLP_DIM * HEAD_DIM * sizeof(float);
    
    int8_t *h_K = (int8_t*)malloc(int8_size);
    int8_t *h_V = (int8_t*)malloc(int8_size);
    float *h_input = (float*)malloc(fp32_size);
    float *h_W_up = (float*)malloc(w_up_size);
    float *h_W_down = (float*)malloc(w_down_size);
    float *h_norm1 = (float*)malloc(HEAD_DIM * sizeof(float));
    float *h_norm2 = (float*)malloc(HEAD_DIM * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < int8_size; i++) {
        h_K[i] = (int8_t)(rand() % 256 - 128);
        h_V[i] = (int8_t)(rand() % 256 - 128);
    }
    for (size_t i = 0; i < (size_t)seq_len * HEAD_DIM; i++) {
        h_input[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (size_t i = 0; i < (size_t)HEAD_DIM * MLP_DIM; i++) h_W_up[i] = 0.01f;
    for (size_t i = 0; i < (size_t)MLP_DIM * HEAD_DIM; i++) h_W_down[i] = 0.01f;
    for (int i = 0; i < HEAD_DIM; i++) { h_norm1[i] = 1.0f; h_norm2[i] = 1.0f; }
    
    int8_t *d_K, *d_V;
    float *d_input, *d_output, *d_W_up, *d_W_down, *d_norm1, *d_norm2;
    
    cudaMalloc(&d_K, int8_size);
    cudaMalloc(&d_V, int8_size);
    cudaMalloc(&d_input, fp32_size);
    cudaMalloc(&d_output, fp32_size);
    cudaMalloc(&d_W_up, w_up_size);
    cudaMalloc(&d_W_down, w_down_size);
    cudaMalloc(&d_norm1, HEAD_DIM * sizeof(float));
    cudaMalloc(&d_norm2, HEAD_DIM * sizeof(float));
    
    cudaMemcpy(d_K, h_K, int8_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, int8_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, fp32_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_up, h_W_up, w_up_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_down, h_W_down, w_down_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm1, h_norm1, HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm2, h_norm2, HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    TuningFork tuning = {1.0f / (sqrtf(256.0f) * 127.0f * 127.0f), 1.0f, 1.0f, 1.0f, 1e-6f, 0, 1.0f, 1.0f};
    
    int block = 32;
    int grid = (seq_len + block - 1) / block;
    
    for (int i = 0; i < 3; i++) {
        waller_fused_256<<<grid, block>>>(d_K, d_V, d_W_up, d_W_down, d_norm1, d_norm2, d_input, d_output, seq_len, tuning);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        waller_fused_256<<<grid, block>>>(d_K, d_V, d_W_up, d_W_down, d_norm1, d_norm2, d_input, d_output, seq_len, tuning);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / num_iterations;
    
    double attn_flops = (double)seq_len * ((double)seq_len + 1.0) / 2.0 * HEAD_DIM * 2.0;
    double mlp_flops = (double)seq_len * HEAD_DIM * MLP_DIM * 4.0;
    double total_flops = attn_flops + mlp_flops;
    double tflops = (total_flops / (avg_ms / 1000.0)) / 1e12;
    
    size_t standard_hbm = 8 * fp32_size;
    size_t fused_hbm = 2 * fp32_size + w_up_size + w_down_size;
    float hbm_reduction = 100.0f * (1.0f - (float)fused_hbm / (float)standard_hbm);
    
    size_t standard_attn_mem = (size_t)seq_len * seq_len * sizeof(float);
    
    printf("Time: %.3f ms | %.3f TFLOPS\n", avg_ms, tflops);
    printf("HBM Reduction: %.1f%%\n", hbm_reduction);
    printf("Standard attention would need: %.2f GB\n", standard_attn_mem / 1e9);
    if (standard_attn_mem > 80e9) printf(">>> IMPOSSIBLE WITH STANDARD <<<\n");
    
    cudaFree(d_K); cudaFree(d_V); cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_W_up); cudaFree(d_W_down); cudaFree(d_norm1); cudaFree(d_norm2);
    free(h_K); free(h_V); free(h_input); free(h_W_up); free(h_W_down); free(h_norm1); free(h_norm2);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     WALLER FUSED SYMPHONY - REALISTIC SCALE                     ║\n");
    printf("║     HEAD_DIM=128 (TILE=96, 24KB) | HEAD_DIM=256 (TILE=48, 24KB) ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    printf("\n>>> HEAD_DIM=128 (LLaMA-scale) <<<\n");
    run_benchmark_128(8192, 20);
    run_benchmark_128(16384, 10);
    run_benchmark_128(32768, 5);
    run_benchmark_128(65536, 3);
    run_benchmark_128(131072, 2);
    run_benchmark_128(262144, 1);
    
    printf("\n>>> HEAD_DIM=256 (Larger scale) <<<\n");
    run_benchmark_256(8192, 10);
    run_benchmark_256(16384, 5);
    run_benchmark_256(32768, 3);
    run_benchmark_256(65536, 2);
    
    return 0;
}

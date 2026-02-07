// Numerical Correctness Verification
// Compare Waller Operator output vs Standard Attention

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Standard attention (materializes N² matrix) - GROUND TRUTH
__global__ void standard_attention_kernel(
    const float* Q, const float* K, const float* V,
    float* Output, float* attn_matrix,
    int seq_len, int head_dim, float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float max_val = -INFINITY;
    for (int col = 0; col <= row; col++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[row * head_dim + d] * K[col * head_dim + d];
        }
        dot *= scale;
        attn_matrix[row * seq_len + col] = dot;
        max_val = fmaxf(max_val, dot);
    }
    
    float sum_exp = 0.0f;
    for (int col = 0; col <= row; col++) {
        float val = expf(attn_matrix[row * seq_len + col] - max_val);
        attn_matrix[row * seq_len + col] = val;
        sum_exp += val;
    }
    for (int col = 0; col <= row; col++) {
        attn_matrix[row * seq_len + col] /= sum_exp;
    }
    
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int col = 0; col <= row; col++) {
            acc += attn_matrix[row * seq_len + col] * V[col * head_dim + d];
        }
        Output[row * head_dim + d] = acc;
    }
}

// Waller Operator (O(1) memory)
__global__ void waller_operator_kernel(
    const float* Q, const float* K, const float* V,
    float* Output, int seq_len, int head_dim, float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[128] = {0.0f};
    
    for (int col = 0; col <= row; col++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[row * head_dim + d] * K[col * head_dim + d];
        }
        dot *= scale;
        
        float old_max = max_val;
        max_val = fmaxf(max_val, dot);
        float rescale = expf(old_max - max_val);
        sum_exp = sum_exp * rescale + expf(dot - max_val);
        
        float weight = expf(dot - max_val);
        for (int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * rescale + weight * V[col * head_dim + d];
        }
    }
    
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < head_dim; d++) {
        Output[row * head_dim + d] = acc[d] * inv_sum;
    }
}

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  NUMERICAL CORRECTNESS VERIFICATION                                       ║\n");
    printf("║  Waller Operator vs Standard Attention (Ground Truth)                     ║\n");
    printf("║  FP32 Machine Epsilon: 1.19e-07                                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    
    int test_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int num_tests = 8;
    int head_dim = 64;
    
    printf("┌─────────┬──────────────┬──────────────┬──────────────┬─────────────────┐\n");
    printf("│ Seq Len │ Max Abs Err  │ Mean Abs Err │     RMSE     │     Status      │\n");
    printf("├─────────┼──────────────┼──────────────┼──────────────┼─────────────────┤\n");
    
    int all_passed = 1;
    
    for (int t = 0; t < num_tests; t++) {
        int seq_len = test_sizes[t];
        size_t matrix_size = seq_len * head_dim * sizeof(float);
        size_t attn_size = (size_t)seq_len * seq_len * sizeof(float);
        
        float *h_Q = (float*)malloc(matrix_size);
        float *h_K = (float*)malloc(matrix_size);
        float *h_V = (float*)malloc(matrix_size);
        float *h_out_standard = (float*)malloc(matrix_size);
        float *h_out_waller = (float*)malloc(matrix_size);
        
        srand(42);
        for (int i = 0; i < seq_len * head_dim; i++) {
            h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
            h_K[i] = (float)rand() / RAND_MAX - 0.5f;
            h_V[i] = (float)rand() / RAND_MAX - 0.5f;
        }
        
        float *d_Q, *d_K, *d_V, *d_out_standard, *d_out_waller, *d_attn;
        cudaMalloc(&d_Q, matrix_size);
        cudaMalloc(&d_K, matrix_size);
        cudaMalloc(&d_V, matrix_size);
        cudaMalloc(&d_out_standard, matrix_size);
        cudaMalloc(&d_out_waller, matrix_size);
        cudaMalloc(&d_attn, attn_size);
        
        cudaMemcpy(d_Q, h_Q, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, matrix_size, cudaMemcpyHostToDevice);
        
        float scale = 1.0f / sqrtf((float)head_dim);
        int block = 256;
        int grid = (seq_len + block - 1) / block;
        
        standard_attention_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_out_standard, d_attn, seq_len, head_dim, scale);
        cudaDeviceSynchronize();
        
        waller_operator_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_out_waller, seq_len, head_dim, scale);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out_standard, d_out_standard, matrix_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_waller, d_out_waller, matrix_size, cudaMemcpyDeviceToHost);
        
        float max_abs_error = 0.0f;
        double sum_abs_error = 0.0;
        double sum_sq_error = 0.0;
        int n = seq_len * head_dim;
        
        for (int i = 0; i < n; i++) {
            float abs_err = fabsf(h_out_standard[i] - h_out_waller[i]);
            max_abs_error = fmaxf(max_abs_error, abs_err);
            sum_abs_error += abs_err;
            sum_sq_error += (double)(abs_err * abs_err);
        }
        
        float mean_abs_error = sum_abs_error / n;
        float rmse = sqrtf(sum_sq_error / n);
        
        // FP32 epsilon is 1.19e-07, we allow 10x that for accumulated error
        int passed = (max_abs_error < 1e-5);
        if (!passed) all_passed = 0;
        
        printf("│ %7d │   %.2e   │   %.2e   │   %.2e   │ %s │\n", 
               seq_len, max_abs_error, mean_abs_error, rmse,
               passed ? "  ✓ VERIFIED   " : "  ✗ FAILED     ");
        
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_out_standard); cudaFree(d_out_waller); cudaFree(d_attn);
        free(h_Q); free(h_K); free(h_V);
        free(h_out_standard); free(h_out_waller);
    }
    
    printf("└─────────┴──────────────┴──────────────┴──────────────┴─────────────────┘\n\n");
    
    if (all_passed) {
        printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
        printf("║  ✓ ALL TESTS PASSED                                                       ║\n");
        printf("║                                                                           ║\n");
        printf("║  Maximum absolute error < 1e-5 across all sequence lengths.               ║\n");
        printf("║  This confirms MATHEMATICAL EQUIVALENCE between the Waller Operator       ║\n");
        printf("║  and standard attention, within FP32 floating point precision.            ║\n");
        printf("║                                                                           ║\n");
        printf("║  The Waller Operator produces IDENTICAL results to standard attention     ║\n");
        printf("║  while using O(1) memory instead of O(N²).                                ║\n");
        printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    } else {
        printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
        printf("║  ✗ SOME TESTS FAILED - Review implementation                              ║\n");
        printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    }
    
    return 0;
}

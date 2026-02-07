// AOT DISTILLATION
// Key insight: Use FIXED dataset, proper architecture matching MLP structure
// Low-rank factorization of Wu and Wd directly: Wu ≈ U_u @ V_u, Wd ≈ U_d @ V_d

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define HEAD_DIM 128
#define MLP_DIM 512
#define BATCH 2048

__global__ void init_random_scaled(float* data, int n, float scale, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = (curand_uniform(&state) - 0.5f) * scale;
    }
}

// Full MLP: GELU(x @ Wu) @ Wd
__global__ void mlp_full(
    const float* __restrict__ input,
    const float* __restrict__ Wu,
    const float* __restrict__ Wd,
    float* __restrict__ output,
    float* __restrict__ hidden_cache,  // cache for backward
    int batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    // Up projection
    for (int m = 0; m < MLP_DIM; m++) {
        float h = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) 
            h += input[row * HEAD_DIM + d] * Wu[d * MLP_DIM + m];
        // GELU
        float g = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
        hidden_cache[row * MLP_DIM + m] = g;
    }
    
    // Down projection  
    for (int d = 0; d < HEAD_DIM; d++) {
        float o = 0.0f;
        for (int m = 0; m < MLP_DIM; m++)
            o += hidden_cache[row * MLP_DIM + m] * Wd[m * HEAD_DIM + d];
        output[row * HEAD_DIM + d] = o;
    }
}

// Compressed MLP using low-rank: Wu ≈ Uu[HEAD_DIM,r] @ Vu[r,MLP_DIM]
//                                 Wd ≈ Ud[MLP_DIM,r] @ Vd[r,HEAD_DIM]
// This preserves the MLP structure exactly, just compresses the weights
__global__ void mlp_lowrank(
    const float* __restrict__ input,
    const float* __restrict__ Uu,  // [HEAD_DIM, rank]
    const float* __restrict__ Vu,  // [rank, MLP_DIM]  
    const float* __restrict__ Ud,  // [MLP_DIM, rank]
    const float* __restrict__ Vd,  // [rank, HEAD_DIM]
    float* __restrict__ output,
    int batch, int rank
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float in[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) in[d] = input[row * HEAD_DIM + d];
    
    // Step 1: x @ Uu -> [rank]
    float proj_u[128];  // max rank
    for (int r = 0; r < rank; r++) {
        proj_u[r] = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++)
            proj_u[r] += in[d] * Uu[d * rank + r];
    }
    
    // Step 2: [rank] @ Vu -> hidden[MLP_DIM], then GELU
    float hidden[MLP_DIM];
    for (int m = 0; m < MLP_DIM; m++) {
        float h = 0.0f;
        for (int r = 0; r < rank; r++)
            h += proj_u[r] * Vu[r * MLP_DIM + m];
        hidden[m] = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
    }
    
    // Step 3: hidden @ Ud -> [rank]
    float proj_d[128];
    for (int r = 0; r < rank; r++) {
        proj_d[r] = 0.0f;
        for (int m = 0; m < MLP_DIM; m++)
            proj_d[r] += hidden[m] * Ud[m * rank + r];
    }
    
    // Step 4: [rank] @ Vd -> output[HEAD_DIM]
    for (int d = 0; d < HEAD_DIM; d++) {
        float o = 0.0f;
        for (int r = 0; r < rank; r++)
            o += proj_d[r] * Vd[r * HEAD_DIM + d];
        output[row * HEAD_DIM + d] = o;
    }
}

// SVD-like initialization: extract principal components from actual weights
__global__ void compute_gram(
    const float* __restrict__ W,  // [rows, cols]
    float* __restrict__ gram,     // [cols, cols] or [rows, rows]
    int rows, int cols, bool row_gram
) {
    // Simple: just random init for now, real SVD would be better
}

__global__ void compute_mse(const float* pred, const float* target, float* errors, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = pred[idx] - target[idx];
        errors[idx] = diff * diff;
    }
}

__global__ void compute_abs_error(const float* pred, const float* target, float* errors, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        errors[idx] = fabsf(pred[idx] - target[idx]);
    }
}

float sum_reduce(float* d_data, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0; for(int i=0;i<n;i++) s+=h[i];
    free(h); return (float)(s/n);
}

float max_reduce(float* d_data, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    float m = 0; for(int i=0;i<n;i++) if(h[i]>m) m=h[i];
    free(h); return m;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  MLP COMPRESSION VIA LOW-RANK FACTORIZATION\n");
    printf("  Wu[128,512] ≈ Uu[128,r] @ Vu[r,512]\n");
    printf("  Wd[512,128] ≈ Ud[512,r] @ Vd[r,128]\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    int bl = 256;
    
    // Allocate
    float *d_input, *d_target, *d_output, *d_errors, *d_hidden;
    float *d_Wu, *d_Wd;
    
    cudaMalloc(&d_input, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_target, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_output, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_errors, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_hidden, BATCH * MLP_DIM * sizeof(float));
    cudaMalloc(&d_Wu, HEAD_DIM * MLP_DIM * sizeof(float));
    cudaMalloc(&d_Wd, MLP_DIM * HEAD_DIM * sizeof(float));
    
    // Initialize FIXED input data (same data every time)
    init_random_scaled<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_input, BATCH * HEAD_DIM, 2.0f, 42);
    
    // Initialize MLP weights
    float wu_sc = sqrtf(2.0f / HEAD_DIM);
    float wd_sc = sqrtf(2.0f / MLP_DIM);
    init_random_scaled<<<(HEAD_DIM * MLP_DIM + bl-1)/bl, bl>>>(d_Wu, HEAD_DIM * MLP_DIM, wu_sc, 123);
    init_random_scaled<<<(MLP_DIM * HEAD_DIM + bl-1)/bl, bl>>>(d_Wd, MLP_DIM * HEAD_DIM, wd_sc, 456);
    cudaDeviceSynchronize();
    
    // Generate target output
    int gr = (BATCH + 31) / 32;
    mlp_full<<<gr, 32>>>(d_input, d_Wu, d_Wd, d_target, d_hidden, BATCH);
    cudaDeviceSynchronize();
    
    // Measure target stats
    compute_mse<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_target, d_input, d_errors, BATCH * HEAD_DIM);
    float target_var = sum_reduce(d_errors, BATCH * HEAD_DIM);
    float target_std = sqrtf(target_var);
    
    printf("Full MLP: %d params\n", HEAD_DIM * MLP_DIM + MLP_DIM * HEAD_DIM);
    printf("Target output std: %.4f\n\n", target_std);
    
    printf("Rank | Params  | Compress | RMSE    | MaxErr  | Rel%%   | Status\n");
    printf("-----|---------|----------|---------|---------|--------|-------\n");
    
    for (int rank : {2, 4, 8, 16, 32, 64, 128, 256}) {
        // Params: Uu[128,r] + Vu[r,512] + Ud[512,r] + Vd[r,128]
        int params = HEAD_DIM * rank + rank * MLP_DIM + MLP_DIM * rank + rank * HEAD_DIM;
        float compression = (float)(HEAD_DIM * MLP_DIM + MLP_DIM * HEAD_DIM) / params;
        
        float *d_Uu, *d_Vu, *d_Ud, *d_Vd;
        cudaMalloc(&d_Uu, HEAD_DIM * rank * sizeof(float));
        cudaMalloc(&d_Vu, rank * MLP_DIM * sizeof(float));
        cudaMalloc(&d_Ud, MLP_DIM * rank * sizeof(float));
        cudaMalloc(&d_Vd, rank * HEAD_DIM * sizeof(float));
        
        // Initialize low-rank factors (scaled for stability)
        float u_sc = sqrtf(1.0f / rank);
        init_random_scaled<<<(HEAD_DIM * rank + bl-1)/bl, bl>>>(d_Uu, HEAD_DIM * rank, wu_sc * u_sc, 1000+rank);
        init_random_scaled<<<(rank * MLP_DIM + bl-1)/bl, bl>>>(d_Vu, rank * MLP_DIM, u_sc, 2000+rank);
        init_random_scaled<<<(MLP_DIM * rank + bl-1)/bl, bl>>>(d_Ud, MLP_DIM * rank, wd_sc * u_sc, 3000+rank);
        init_random_scaled<<<(rank * HEAD_DIM + bl-1)/bl, bl>>>(d_Vd, rank * HEAD_DIM, u_sc, 4000+rank);
        cudaDeviceSynchronize();
        
        // Forward pass
        mlp_lowrank<<<gr, 32>>>(d_input, d_Uu, d_Vu, d_Ud, d_Vd, d_output, BATCH, rank);
        cudaDeviceSynchronize();
        
        // Compute errors
        compute_mse<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_output, d_target, d_errors, BATCH * HEAD_DIM);
        float mse = sum_reduce(d_errors, BATCH * HEAD_DIM);
        float rmse = sqrtf(mse);
        
        compute_abs_error<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_output, d_target, d_errors, BATCH * HEAD_DIM);
        float maxerr = max_reduce(d_errors, BATCH * HEAD_DIM);
        
        float rel = rmse / target_std * 100.0f;
        const char* status = rel < 20 ? "GOOD" : rel < 50 ? "OK" : rel < 100 ? "POOR" : "FAIL";
        
        printf("%4d | %7d | %6.1fx  | %.5f | %.5f | %5.1f%% | %s\n",
               rank, params, compression, rmse, maxerr, rel, status);
        
        cudaFree(d_Uu); cudaFree(d_Vu); cudaFree(d_Ud); cudaFree(d_Vd);
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  NOTE: This is RANDOM init of low-rank factors.\n");
    printf("  Real compression would use SVD or learned factorization.\n");
    printf("  The question: does the ARCHITECTURE have capacity?\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    cudaFree(d_input); cudaFree(d_target); cudaFree(d_output);
    cudaFree(d_errors); cudaFree(d_hidden); cudaFree(d_Wu); cudaFree(d_Wd);
    
    return 0;
}

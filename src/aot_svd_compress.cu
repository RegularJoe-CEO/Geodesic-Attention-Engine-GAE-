// AOT SVD COMPRESSION
// Actually decompose the MLP weights using SVD
// See the TRUE compression limit

#include <cuda_runtime.h>
#include <cusolverDn.h>
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

// Full MLP
__global__ void mlp_full(
    const float* __restrict__ input,
    const float* __restrict__ Wu,
    const float* __restrict__ Wd,
    float* __restrict__ output,
    int batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float hidden[MLP_DIM];
    for (int m = 0; m < MLP_DIM; m++) {
        float h = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) 
            h += input[row * HEAD_DIM + d] * Wu[d * MLP_DIM + m];
        hidden[m] = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
    }
    
    for (int d = 0; d < HEAD_DIM; d++) {
        float o = 0.0f;
        for (int m = 0; m < MLP_DIM; m++)
            o += hidden[m] * Wd[m * HEAD_DIM + d];
        output[row * HEAD_DIM + d] = o;
    }
}

// Truncated SVD MLP: Wu_truncated = U[:,:r] @ S[:r] @ Vt[:r,:]
__global__ void mlp_svd_truncated(
    const float* __restrict__ input,
    const float* __restrict__ Uu,   // [HEAD_DIM, rank] = U[:,:r] @ sqrt(S[:r])
    const float* __restrict__ Vu,   // [rank, MLP_DIM] = sqrt(S[:r]) @ Vt[:r,:]
    const float* __restrict__ Ud,   // [MLP_DIM, rank]
    const float* __restrict__ Vd,   // [rank, HEAD_DIM]
    float* __restrict__ output,
    int batch, int rank
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float in[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) in[d] = input[row * HEAD_DIM + d];
    
    // x @ Uu @ Vu -> hidden (with GELU)
    float proj_u[256];
    for (int r = 0; r < rank; r++) {
        proj_u[r] = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++)
            proj_u[r] += in[d] * Uu[d * rank + r];
    }
    
    float hidden[MLP_DIM];
    for (int m = 0; m < MLP_DIM; m++) {
        float h = 0.0f;
        for (int r = 0; r < rank; r++)
            h += proj_u[r] * Vu[r * MLP_DIM + m];
        hidden[m] = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
    }
    
    // hidden @ Ud @ Vd -> output
    float proj_d[256];
    for (int r = 0; r < rank; r++) {
        proj_d[r] = 0.0f;
        for (int m = 0; m < MLP_DIM; m++)
            proj_d[r] += hidden[m] * Ud[m * rank + r];
    }
    
    for (int d = 0; d < HEAD_DIM; d++) {
        float o = 0.0f;
        for (int r = 0; r < rank; r++)
            o += proj_d[r] * Vd[r * HEAD_DIM + d];
        output[row * HEAD_DIM + d] = o;
    }
}

void svd_and_truncate(
    float* h_W,      // [rows, cols] input matrix (host)
    float* h_U_out,  // [rows, rank] output (host)
    float* h_V_out,  // [rank, cols] output (host)
    int rows, int cols, int rank
) {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    
    float *d_W, *d_U, *d_S, *d_Vt;
    int *d_info;
    float *d_work;
    int lwork;
    
    cudaMalloc(&d_W, rows * cols * sizeof(float));
    cudaMalloc(&d_U, rows * rows * sizeof(float));
    cudaMalloc(&d_S, min(rows, cols) * sizeof(float));
    cudaMalloc(&d_Vt, cols * cols * sizeof(float));
    cudaMalloc(&d_info, sizeof(int));
    
    cudaMemcpy(d_W, h_W, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Query workspace
    cusolverDnSgesvd_bufferSize(handle, rows, cols, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(float));
    
    // SVD: W = U @ S @ Vt
    cusolverDnSgesvd(handle, 'A', 'A', rows, cols, d_W, rows,
                     d_S, d_U, rows, d_Vt, cols, d_work, lwork, NULL, d_info);
    cudaDeviceSynchronize();
    
    // Copy back and truncate
    float* h_U = (float*)malloc(rows * rows * sizeof(float));
    float* h_S = (float*)malloc(min(rows,cols) * sizeof(float));
    float* h_Vt = (float*)malloc(cols * cols * sizeof(float));
    
    cudaMemcpy(h_U, d_U, rows * rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_S, d_S, min(rows,cols) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Vt, d_Vt, cols * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    // U_out = U[:,:rank] @ sqrt(diag(S[:rank]))
    // V_out = sqrt(diag(S[:rank])) @ Vt[:rank,:]
    for (int i = 0; i < rows; i++) {
        for (int r = 0; r < rank; r++) {
            h_U_out[i * rank + r] = h_U[r * rows + i] * sqrtf(h_S[r]);
        }
    }
    for (int r = 0; r < rank; r++) {
        for (int j = 0; j < cols; j++) {
            h_V_out[r * cols + j] = sqrtf(h_S[r]) * h_Vt[r * cols + j];
        }
    }
    
    // Print singular value spectrum
    if (rows == HEAD_DIM && cols == MLP_DIM) {
        printf("  Wu singular values: ");
        for (int i = 0; i < min(8, min(rows,cols)); i++) printf("%.3f ", h_S[i]);
        printf("...\n");
    }
    
    free(h_U); free(h_S); free(h_Vt);
    cudaFree(d_W); cudaFree(d_U); cudaFree(d_S); cudaFree(d_Vt);
    cudaFree(d_info); cudaFree(d_work);
    cusolverDnDestroy(handle);
}

__global__ void compute_mse(const float* pred, const float* target, float* errors, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = pred[idx] - target[idx];
        errors[idx] = diff * diff;
    }
}

float sum_reduce(float* d_data, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0; for(int i=0;i<n;i++) s+=h[i];
    free(h); return (float)(s/n);
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  SVD-BASED MLP COMPRESSION\n");
    printf("  Actual truncated SVD of weight matrices\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    int bl = 256;
    
    // Host allocations for SVD
    float* h_Wu = (float*)malloc(HEAD_DIM * MLP_DIM * sizeof(float));
    float* h_Wd = (float*)malloc(MLP_DIM * HEAD_DIM * sizeof(float));
    
    // Device allocations
    float *d_input, *d_target, *d_output, *d_errors;
    float *d_Wu, *d_Wd;
    
    cudaMalloc(&d_input, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_target, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_output, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_errors, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_Wu, HEAD_DIM * MLP_DIM * sizeof(float));
    cudaMalloc(&d_Wd, MLP_DIM * HEAD_DIM * sizeof(float));
    
    // Initialize
    init_random_scaled<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_input, BATCH * HEAD_DIM, 2.0f, 42);
    float wu_sc = sqrtf(2.0f / HEAD_DIM);
    float wd_sc = sqrtf(2.0f / MLP_DIM);
    init_random_scaled<<<(HEAD_DIM * MLP_DIM + bl-1)/bl, bl>>>(d_Wu, HEAD_DIM * MLP_DIM, wu_sc, 123);
    init_random_scaled<<<(MLP_DIM * HEAD_DIM + bl-1)/bl, bl>>>(d_Wd, MLP_DIM * HEAD_DIM, wd_sc, 456);
    cudaDeviceSynchronize();
    
    // Copy weights to host for SVD
    cudaMemcpy(h_Wu, d_Wu, HEAD_DIM * MLP_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Wd, d_Wd, MLP_DIM * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Generate target
    int gr = (BATCH + 31) / 32;
    mlp_full<<<gr, 32>>>(d_input, d_Wu, d_Wd, d_target, BATCH);
    cudaDeviceSynchronize();
    
    // Get target magnitude
    compute_mse<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_target, d_input, d_errors, BATCH * HEAD_DIM);
    float target_std = sqrtf(sum_reduce(d_errors, BATCH * HEAD_DIM));
    
    printf("Full MLP: %d params, output std: %.4f\n\n", HEAD_DIM * MLP_DIM + MLP_DIM * HEAD_DIM, target_std);
    
    printf("Rank | Params  | Compress | RMSE     | Rel%%    | Quality\n");
    printf("-----|---------|----------|----------|---------|--------\n");
    
    for (int rank : {1, 2, 4, 8, 16, 32, 64, 128}) {
        int params = HEAD_DIM * rank + rank * MLP_DIM + MLP_DIM * rank + rank * HEAD_DIM;
        float compression = (float)(HEAD_DIM * MLP_DIM + MLP_DIM * HEAD_DIM) / params;
        
        // SVD decomposition
        float* h_Uu = (float*)malloc(HEAD_DIM * rank * sizeof(float));
        float* h_Vu = (float*)malloc(rank * MLP_DIM * sizeof(float));
        float* h_Ud = (float*)malloc(MLP_DIM * rank * sizeof(float));
        float* h_Vd = (float*)malloc(rank * HEAD_DIM * sizeof(float));
        
        if (rank == 1) {
            printf("\nSVD spectrum analysis:\n");
        }
        svd_and_truncate(h_Wu, h_Uu, h_Vu, HEAD_DIM, MLP_DIM, rank);
        svd_and_truncate(h_Wd, h_Ud, h_Vd, MLP_DIM, HEAD_DIM, rank);
        if (rank == 1) printf("\n");
        
        // Upload to device
        float *d_Uu, *d_Vu, *d_Ud, *d_Vd;
        cudaMalloc(&d_Uu, HEAD_DIM * rank * sizeof(float));
        cudaMalloc(&d_Vu, rank * MLP_DIM * sizeof(float));
        cudaMalloc(&d_Ud, MLP_DIM * rank * sizeof(float));
        cudaMalloc(&d_Vd, rank * HEAD_DIM * sizeof(float));
        
        cudaMemcpy(d_Uu, h_Uu, HEAD_DIM * rank * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vu, h_Vu, rank * MLP_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ud, h_Ud, MLP_DIM * rank * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vd, h_Vd, rank * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
        
        // Forward
        mlp_svd_truncated<<<gr, 32>>>(d_input, d_Uu, d_Vu, d_Ud, d_Vd, d_output, BATCH, rank);
        cudaDeviceSynchronize();
        
        // Measure error
        compute_mse<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_output, d_target, d_errors, BATCH * HEAD_DIM);
        float rmse = sqrtf(sum_reduce(d_errors, BATCH * HEAD_DIM));
        float rel = rmse / target_std * 100.0f;
        
        const char* status = rel < 5 ? "GREAT" : rel < 20 ? "GOOD" : rel < 50 ? "OK" : "POOR";
        
        printf("%4d | %7d | %6.1fx  | %.6f | %6.2f%% | %s\n",
               rank, params, compression, rmse, rel, status);
        
        free(h_Uu); free(h_Vu); free(h_Ud); free(h_Vd);
        cudaFree(d_Uu); cudaFree(d_Vu); cudaFree(d_Ud); cudaFree(d_Vd);
    }
    
    free(h_Wu); free(h_Wd);
    cudaFree(d_input); cudaFree(d_target); cudaFree(d_output);
    cudaFree(d_errors); cudaFree(d_Wu); cudaFree(d_Wd);
    
    return 0;
}

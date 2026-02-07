// Fixed SVD - handle column-major properly
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

__global__ void mlp_lowrank_direct(
    const float* __restrict__ input,
    const float* __restrict__ Wu_compressed,  // [HEAD_DIM, rank]
    const float* __restrict__ Wv_compressed,  // [rank, MLP_DIM]
    const float* __restrict__ Wd,             // Full Wd for now
    float* __restrict__ output,
    int batch, int rank
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float in[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) in[d] = input[row * HEAD_DIM + d];
    
    // x @ Wu_compressed -> [rank]
    float proj[256];
    for (int r = 0; r < rank; r++) {
        proj[r] = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++)
            proj[r] += in[d] * Wu_compressed[d * rank + r];
    }
    
    // [rank] @ Wv_compressed -> hidden[MLP_DIM] + GELU
    float hidden[MLP_DIM];
    for (int m = 0; m < MLP_DIM; m++) {
        float h = 0.0f;
        for (int r = 0; r < rank; r++)
            h += proj[r] * Wv_compressed[r * MLP_DIM + m];
        hidden[m] = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
    }
    
    // hidden @ Wd -> output
    for (int d = 0; d < HEAD_DIM; d++) {
        float o = 0.0f;
        for (int m = 0; m < MLP_DIM; m++)
            o += hidden[m] * Wd[m * HEAD_DIM + d];
        output[row * HEAD_DIM + d] = o;
    }
}

// CPU SVD using simple power iteration (avoid cuSOLVER complexity)
void cpu_svd_truncated(
    float* W,      // [rows, cols] row-major
    float* U_out,  // [rows, rank] 
    float* V_out,  // [rank, cols]
    float* S_out,  // [rank] singular values
    int rows, int cols, int rank
) {
    float* W_work = (float*)malloc(rows * cols * sizeof(float));
    memcpy(W_work, W, rows * cols * sizeof(float));
    
    float* u = (float*)malloc(rows * sizeof(float));
    float* v = (float*)malloc(cols * sizeof(float));
    
    for (int r = 0; r < rank; r++) {
        // Initialize v randomly
        for (int j = 0; j < cols; j++) v[j] = (float)rand() / RAND_MAX - 0.5f;
        
        // Power iteration
        for (int iter = 0; iter < 100; iter++) {
            // u = W @ v
            float u_norm = 0.0f;
            for (int i = 0; i < rows; i++) {
                u[i] = 0.0f;
                for (int j = 0; j < cols; j++)
                    u[i] += W_work[i * cols + j] * v[j];
                u_norm += u[i] * u[i];
            }
            u_norm = sqrtf(u_norm);
            for (int i = 0; i < rows; i++) u[i] /= (u_norm + 1e-10f);
            
            // v = W^T @ u
            float v_norm = 0.0f;
            for (int j = 0; j < cols; j++) {
                v[j] = 0.0f;
                for (int i = 0; i < rows; i++)
                    v[j] += W_work[i * cols + j] * u[i];
                v_norm += v[j] * v[j];
            }
            v_norm = sqrtf(v_norm);
            for (int j = 0; j < cols; j++) v[j] /= (v_norm + 1e-10f);
        }
        
        // Compute singular value: sigma = u^T @ W @ v
        float sigma = 0.0f;
        for (int i = 0; i < rows; i++) {
            float wv = 0.0f;
            for (int j = 0; j < cols; j++)
                wv += W_work[i * cols + j] * v[j];
            sigma += u[i] * wv;
        }
        
        S_out[r] = sigma;
        
        // Store U[:, r] = u * sqrt(sigma), V[r, :] = v * sqrt(sigma)
        float sqrt_s = sqrtf(fabsf(sigma));
        for (int i = 0; i < rows; i++)
            U_out[i * rank + r] = u[i] * sqrt_s;
        for (int j = 0; j < cols; j++)
            V_out[r * cols + j] = v[j] * sqrt_s;
        
        // Deflate: W = W - sigma * u @ v^T
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                W_work[i * cols + j] -= sigma * u[i] * v[j];
    }
    
    free(W_work);
    free(u);
    free(v);
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
    printf("  SVD MLP COMPRESSION (CPU Power Iteration)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    srand(42);
    int bl = 256;
    
    // Allocate
    float* h_Wu = (float*)malloc(HEAD_DIM * MLP_DIM * sizeof(float));
    float* h_Wd = (float*)malloc(MLP_DIM * HEAD_DIM * sizeof(float));
    
    float *d_input, *d_target, *d_output, *d_errors;
    float *d_Wu, *d_Wd;
    
    cudaMalloc(&d_input, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_target, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_output, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_errors, BATCH * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_Wu, HEAD_DIM * MLP_DIM * sizeof(float));
    cudaMalloc(&d_Wd, MLP_DIM * HEAD_DIM * sizeof(float));
    
    init_random_scaled<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_input, BATCH * HEAD_DIM, 2.0f, 42);
    float wu_sc = sqrtf(2.0f / HEAD_DIM);
    float wd_sc = sqrtf(2.0f / MLP_DIM);
    init_random_scaled<<<(HEAD_DIM * MLP_DIM + bl-1)/bl, bl>>>(d_Wu, HEAD_DIM * MLP_DIM, wu_sc, 123);
    init_random_scaled<<<(MLP_DIM * HEAD_DIM + bl-1)/bl, bl>>>(d_Wd, MLP_DIM * HEAD_DIM, wd_sc, 456);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_Wu, d_Wu, HEAD_DIM * MLP_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Wd, d_Wd, MLP_DIM * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    
    int gr = (BATCH + 31) / 32;
    mlp_full<<<gr, 32>>>(d_input, d_Wu, d_Wd, d_target, BATCH);
    cudaDeviceSynchronize();
    
    compute_mse<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_target, d_input, d_errors, BATCH * HEAD_DIM);
    float target_std = sqrtf(sum_reduce(d_errors, BATCH * HEAD_DIM));
    
    printf("Full MLP: %d params, output std: %.4f\n\n", HEAD_DIM * MLP_DIM + MLP_DIM * HEAD_DIM, target_std);
    
    // First: just compress Wu, keep Wd full
    printf("=== Compressing Wu only (keep Wd full) ===\n");
    printf("Rank | Wu Params | Singular Values (top 4)        | RMSE     | Rel%%\n");
    printf("-----|-----------|--------------------------------|----------|------\n");
    
    for (int rank : {1, 2, 4, 8, 16, 32, 64, 128}) {
        float* h_Uu = (float*)malloc(HEAD_DIM * rank * sizeof(float));
        float* h_Vu = (float*)malloc(rank * MLP_DIM * sizeof(float));
        float* h_Su = (float*)malloc(rank * sizeof(float));
        
        cpu_svd_truncated(h_Wu, h_Uu, h_Vu, h_Su, HEAD_DIM, MLP_DIM, rank);
        
        // Upload
        float *d_Uu, *d_Vu;
        cudaMalloc(&d_Uu, HEAD_DIM * rank * sizeof(float));
        cudaMalloc(&d_Vu, rank * MLP_DIM * sizeof(float));
        cudaMemcpy(d_Uu, h_Uu, HEAD_DIM * rank * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vu, h_Vu, rank * MLP_DIM * sizeof(float), cudaMemcpyHostToDevice);
        
        // Forward with compressed Wu
        mlp_lowrank_direct<<<gr, 32>>>(d_input, d_Uu, d_Vu, d_Wd, d_output, BATCH, rank);
        cudaDeviceSynchronize();
        
        compute_mse<<<(BATCH * HEAD_DIM + bl-1)/bl, bl>>>(d_output, d_target, d_errors, BATCH * HEAD_DIM);
        float rmse = sqrtf(sum_reduce(d_errors, BATCH * HEAD_DIM));
        float rel = rmse / target_std * 100.0f;
        
        int wu_params = HEAD_DIM * rank + rank * MLP_DIM;
        printf("%4d | %9d | %6.3f %6.3f %6.3f %6.3f | %.6f | %5.1f%%\n",
               rank, wu_params, 
               h_Su[0], rank > 1 ? h_Su[1] : 0.0f, 
               rank > 2 ? h_Su[2] : 0.0f, rank > 3 ? h_Su[3] : 0.0f,
               rmse, rel);
        
        free(h_Uu); free(h_Vu); free(h_Su);
        cudaFree(d_Uu); cudaFree(d_Vu);
        
        // Reinit h_Wu (it gets modified by SVD)
        cudaMemcpy(h_Wu, d_Wu, HEAD_DIM * MLP_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    free(h_Wu); free(h_Wd);
    cudaFree(d_input); cudaFree(d_target); cudaFree(d_output);
    cudaFree(d_errors); cudaFree(d_Wu); cudaFree(d_Wd);
    
    return 0;
}

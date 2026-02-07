// AOT CAPABILITY TEST
// Can compressed shapes actually approximate what a real MLP does?
// We'll create a "ground truth" MLP and try to fit our shapes to match it

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define HEAD_DIM 128
#define MLP_DIM 512

// Initialize random weights
__global__ void init_random(float* data, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = (curand_uniform(&state) - 0.5f) * 0.1f;
    }
}

// Ground truth MLP: Wu[128,512] -> GELU -> Wd[512,128]
__global__ void mlp_ground_truth(
    const float* __restrict__ input,   // [batch, HEAD_DIM]
    const float* __restrict__ Wu,      // [HEAD_DIM, MLP_DIM]
    const float* __restrict__ Wd,      // [MLP_DIM, HEAD_DIM]
    float* __restrict__ output,        // [batch, HEAD_DIM]
    int batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float in[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) in[d] = input[row * HEAD_DIM + d];
    
    // Up projection + GELU
    float hidden[MLP_DIM];
    for (int m = 0; m < MLP_DIM; m++) {
        float h = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) h += in[d] * Wu[d * MLP_DIM + m];
        hidden[m] = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
    }
    
    // Down projection
    for (int d = 0; d < HEAD_DIM; d++) {
        float o = 0.0f;
        for (int m = 0; m < MLP_DIM; m++) o += hidden[m] * Wd[m * HEAD_DIM + d];
        output[row * HEAD_DIM + d] = o;
    }
}

// SHAPE 1: Minimal (scale + bias)
__global__ void mlp_minimal(
    const float* __restrict__ input,
    const float* __restrict__ scale,   // [HEAD_DIM]
    const float* __restrict__ bias,    // [HEAD_DIM]
    float* __restrict__ output,
    int batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    for (int d = 0; d < HEAD_DIM; d++) {
        float x = input[row * HEAD_DIM + d] * scale[d] + bias[d];
        output[row * HEAD_DIM + d] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// SHAPE 2: Conv kernel
__global__ void mlp_conv(
    const float* __restrict__ input,
    const float* __restrict__ kernel,  // [kernel_size]
    float* __restrict__ output,
    int batch, int kernel_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    int half_k = kernel_size / 2;
    for (int d = 0; d < HEAD_DIM; d++) {
        float conv = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int src = d - half_k + k;
            if (src >= 0 && src < HEAD_DIM) {
                conv += input[row * HEAD_DIM + src] * kernel[k];
            }
        }
        output[row * HEAD_DIM + d] = 0.5f * conv * (1.0f + tanhf(0.7978845608f * (conv + 0.044715f * conv * conv * conv)));
    }
}

// SHAPE 3: Rank-R approximation
__global__ void mlp_rank(
    const float* __restrict__ input,
    const float* __restrict__ shape_a,  // [R, HEAD_DIM]
    const float* __restrict__ shape_b,  // [R, HEAD_DIM]
    float* __restrict__ output,
    int batch, int rank
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float out[HEAD_DIM] = {0};
    for (int r = 0; r < rank; r++) {
        float proj = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) proj += input[row * HEAD_DIM + d] * shape_a[r * HEAD_DIM + d];
        proj = 0.5f * proj * (1.0f + tanhf(0.7978845608f * (proj + 0.044715f * proj * proj * proj)));
        for (int d = 0; d < HEAD_DIM; d++) out[d] += proj * shape_b[r * HEAD_DIM + d];
    }
    
    for (int d = 0; d < HEAD_DIM; d++) output[row * HEAD_DIM + d] = out[d];
}

// Compute MSE between two outputs
__global__ void compute_mse(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ errors,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = pred[idx] - target[idx];
        errors[idx] = diff * diff;
    }
}

float sum_reduce(float* d_data, int n) {
    float* h_data = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += h_data[i];
    free(h_data);
    return (float)(sum / n);
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  CAPABILITY TEST: Can shapes approximate a real MLP?\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    int batch = 4096;  // Test vectors
    size_t io_size = batch * HEAD_DIM * sizeof(float);
    size_t wu_size = HEAD_DIM * MLP_DIM * sizeof(float);
    size_t wd_size = MLP_DIM * HEAD_DIM * sizeof(float);
    
    // Allocate
    float *d_input, *d_gt_output, *d_test_output, *d_errors;
    float *d_Wu, *d_Wd;
    cudaMalloc(&d_input, io_size);
    cudaMalloc(&d_gt_output, io_size);
    cudaMalloc(&d_test_output, io_size);
    cudaMalloc(&d_errors, io_size);
    cudaMalloc(&d_Wu, wu_size);
    cudaMalloc(&d_Wd, wd_size);
    
    // Initialize random input and MLP weights
    int bl = 256;
    init_random<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_input, batch * HEAD_DIM, 42);
    init_random<<<(HEAD_DIM * MLP_DIM + bl - 1) / bl, bl>>>(d_Wu, HEAD_DIM * MLP_DIM, 123);
    init_random<<<(MLP_DIM * HEAD_DIM + bl - 1) / bl, bl>>>(d_Wd, MLP_DIM * HEAD_DIM, 456);
    cudaDeviceSynchronize();
    
    // Generate ground truth output
    int gr = (batch + 31) / 32;
    mlp_ground_truth<<<gr, 32>>>(d_input, d_Wu, d_Wd, d_gt_output, batch);
    cudaDeviceSynchronize();
    
    // Compute ground truth output magnitude (for relative error)
    compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_gt_output, d_input, d_errors, batch * HEAD_DIM);
    float gt_variance = sum_reduce(d_errors, batch * HEAD_DIM);
    
    printf("Ground truth MLP: 131,072 parameters\n");
    printf("Test batch: %d vectors of dim %d\n", batch, HEAD_DIM);
    printf("GT output variance: %.6f\n\n", gt_variance);
    
    printf("Approach         | Params  | MSE        | Relative | Quality\n");
    printf("-----------------|---------|------------|----------|--------\n");
    
    // Test MINIMAL (random init)
    float *d_scale, *d_bias;
    cudaMalloc(&d_scale, HEAD_DIM * sizeof(float));
    cudaMalloc(&d_bias, HEAD_DIM * sizeof(float));
    init_random<<<1, HEAD_DIM>>>(d_scale, HEAD_DIM, 789);
    init_random<<<1, HEAD_DIM>>>(d_bias, HEAD_DIM, 101);
    cudaDeviceSynchronize();
    
    mlp_minimal<<<gr, 32>>>(d_input, d_scale, d_bias, d_test_output, batch);
    compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_test_output, d_gt_output, d_errors, batch * HEAD_DIM);
    float mse_min = sum_reduce(d_errors, batch * HEAD_DIM);
    float rel_min = mse_min / gt_variance * 100.0f;
    printf("MINIMAL (s+b)    | %7d | %.6f | %6.1f%%  | %s\n", 
           256, mse_min, rel_min, rel_min < 10 ? "GOOD" : rel_min < 50 ? "OK" : "POOR");
    cudaFree(d_scale); cudaFree(d_bias);
    
    // Test CONV kernels
    for (int ks : {3, 7, 15, 31}) {
        float *d_kernel;
        cudaMalloc(&d_kernel, ks * sizeof(float));
        init_random<<<1, ks>>>(d_kernel, ks, 200 + ks);
        cudaDeviceSynchronize();
        
        mlp_conv<<<gr, 32>>>(d_input, d_kernel, d_test_output, batch, ks);
        compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_test_output, d_gt_output, d_errors, batch * HEAD_DIM);
        float mse = sum_reduce(d_errors, batch * HEAD_DIM);
        float rel = mse / gt_variance * 100.0f;
        printf("CONV kernel=%-3d  | %7d | %.6f | %6.1f%%  | %s\n",
               ks, ks, mse, rel, rel < 10 ? "GOOD" : rel < 50 ? "OK" : "POOR");
        cudaFree(d_kernel);
    }
    
    // Test RANK approximations
    for (int r : {1, 4, 16, 64, 128}) {
        float *d_a, *d_b;
        cudaMalloc(&d_a, r * HEAD_DIM * sizeof(float));
        cudaMalloc(&d_b, r * HEAD_DIM * sizeof(float));
        init_random<<<(r * HEAD_DIM + bl - 1) / bl, bl>>>(d_a, r * HEAD_DIM, 300 + r);
        init_random<<<(r * HEAD_DIM + bl - 1) / bl, bl>>>(d_b, r * HEAD_DIM, 400 + r);
        cudaDeviceSynchronize();
        
        mlp_rank<<<gr, 32>>>(d_input, d_a, d_b, d_test_output, batch, r);
        compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_test_output, d_gt_output, d_errors, batch * HEAD_DIM);
        float mse = sum_reduce(d_errors, batch * HEAD_DIM);
        float rel = mse / gt_variance * 100.0f;
        int params = r * HEAD_DIM * 2;
        printf("RANK-%-3d         | %7d | %.6f | %6.1f%%  | %s\n",
               r, params, mse, rel, rel < 10 ? "GOOD" : rel < 50 ? "OK" : "POOR");
        cudaFree(d_a); cudaFree(d_b);
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  NOTE: These are RANDOM init, not trained/fitted shapes.\n");
    printf("  High error is expected. Real test = can we LEARN these shapes?\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    cudaFree(d_input); cudaFree(d_gt_output); cudaFree(d_test_output);
    cudaFree(d_errors); cudaFree(d_Wu); cudaFree(d_Wd);
    
    return 0;
}

// AOT CAPABILITY TEST - HARD MODE
// MLP with real-scale weights that actually transforms the input significantly

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define HEAD_DIM 128
#define MLP_DIM 512

__global__ void init_random_scaled(float* data, int n, float scale, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = (curand_uniform(&state) - 0.5f) * scale;
    }
}

__global__ void mlp_ground_truth(
    const float* __restrict__ input,
    const float* __restrict__ Wu,
    const float* __restrict__ Wd,
    float* __restrict__ output,
    int batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float in[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) in[d] = input[row * HEAD_DIM + d];
    
    float hidden[MLP_DIM];
    for (int m = 0; m < MLP_DIM; m++) {
        float h = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) h += in[d] * Wu[d * MLP_DIM + m];
        hidden[m] = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
    }
    
    for (int d = 0; d < HEAD_DIM; d++) {
        float o = 0.0f;
        for (int m = 0; m < MLP_DIM; m++) o += hidden[m] * Wd[m * HEAD_DIM + d];
        output[row * HEAD_DIM + d] = o;
    }
}

__global__ void mlp_minimal(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
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

__global__ void mlp_conv(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
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

__global__ void mlp_rank(
    const float* __restrict__ input,
    const float* __restrict__ shape_a,
    const float* __restrict__ shape_b,
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

__global__ void compute_magnitude(const float* data, float* mag, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) mag[idx] = data[idx] * data[idx];
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
    printf("  CAPABILITY TEST - HARD MODE\n");
    printf("  MLP weights scaled to do REAL transformation\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    int batch = 4096;
    size_t io_size = batch * HEAD_DIM * sizeof(float);
    size_t wu_size = HEAD_DIM * MLP_DIM * sizeof(float);
    size_t wd_size = MLP_DIM * HEAD_DIM * sizeof(float);
    
    float *d_input, *d_gt_output, *d_test_output, *d_errors;
    float *d_Wu, *d_Wd;
    cudaMalloc(&d_input, io_size);
    cudaMalloc(&d_gt_output, io_size);
    cudaMalloc(&d_test_output, io_size);
    cudaMalloc(&d_errors, io_size);
    cudaMalloc(&d_Wu, wu_size);
    cudaMalloc(&d_Wd, wd_size);
    
    int bl = 256;
    
    // Input: unit scale random
    init_random_scaled<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_input, batch * HEAD_DIM, 2.0f, 42);
    
    // MLP weights: Xavier-like init = sqrt(2/fan_in)
    // Wu: fan_in = 128, scale ~ 0.125
    // Wd: fan_in = 512, scale ~ 0.0625
    float wu_scale = sqrtf(2.0f / HEAD_DIM);   // ~0.125
    float wd_scale = sqrtf(2.0f / MLP_DIM);    // ~0.0625
    
    init_random_scaled<<<(HEAD_DIM * MLP_DIM + bl - 1) / bl, bl>>>(d_Wu, HEAD_DIM * MLP_DIM, wu_scale, 123);
    init_random_scaled<<<(MLP_DIM * HEAD_DIM + bl - 1) / bl, bl>>>(d_Wd, MLP_DIM * HEAD_DIM, wd_scale, 456);
    cudaDeviceSynchronize();
    
    int gr = (batch + 31) / 32;
    mlp_ground_truth<<<gr, 32>>>(d_input, d_Wu, d_Wd, d_gt_output, batch);
    cudaDeviceSynchronize();
    
    // Measure input magnitude
    compute_magnitude<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_input, d_errors, batch * HEAD_DIM);
    float input_mag = sqrtf(sum_reduce(d_errors, batch * HEAD_DIM));
    
    // Measure output magnitude  
    compute_magnitude<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_gt_output, d_errors, batch * HEAD_DIM);
    float output_mag = sqrtf(sum_reduce(d_errors, batch * HEAD_DIM));
    
    printf("Input RMS:  %.4f\n", input_mag);
    printf("Output RMS: %.4f (MLP amplification: %.2fx)\n\n", output_mag, output_mag/input_mag);
    
    printf("Approach         | Params  | MSE        | RMSE     | vs Output | Quality\n");
    printf("-----------------|---------|------------|----------|-----------|--------\n");
    
    // MINIMAL
    float *d_scale, *d_bias;
    cudaMalloc(&d_scale, HEAD_DIM * sizeof(float));
    cudaMalloc(&d_bias, HEAD_DIM * sizeof(float));
    init_random_scaled<<<1, HEAD_DIM>>>(d_scale, HEAD_DIM, 1.0f, 789);
    init_random_scaled<<<1, HEAD_DIM>>>(d_bias, HEAD_DIM, 0.1f, 101);
    cudaDeviceSynchronize();
    
    mlp_minimal<<<gr, 32>>>(d_input, d_scale, d_bias, d_test_output, batch);
    compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_test_output, d_gt_output, d_errors, batch * HEAD_DIM);
    float mse = sum_reduce(d_errors, batch * HEAD_DIM);
    float rmse = sqrtf(mse);
    float rel = rmse / output_mag * 100.0f;
    printf("MINIMAL (s+b)    | %7d | %.6f | %.4f   | %6.1f%%   | %s\n", 
           256, mse, rmse, rel, rel < 20 ? "GOOD" : rel < 50 ? "OK" : "POOR");
    cudaFree(d_scale); cudaFree(d_bias);
    
    // CONV kernels
    for (int ks : {3, 7, 15, 31, 63}) {
        float *d_kernel;
        cudaMalloc(&d_kernel, ks * sizeof(float));
        init_random_scaled<<<1, 256>>>(d_kernel, ks, 1.0f/sqrtf(ks), 200 + ks);
        cudaDeviceSynchronize();
        
        mlp_conv<<<gr, 32>>>(d_input, d_kernel, d_test_output, batch, ks);
        compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_test_output, d_gt_output, d_errors, batch * HEAD_DIM);
        mse = sum_reduce(d_errors, batch * HEAD_DIM);
        rmse = sqrtf(mse);
        rel = rmse / output_mag * 100.0f;
        printf("CONV kernel=%-3d  | %7d | %.6f | %.4f   | %6.1f%%   | %s\n",
               ks, ks, mse, rmse, rel, rel < 20 ? "GOOD" : rel < 50 ? "OK" : "POOR");
        cudaFree(d_kernel);
    }
    
    // RANK approximations
    for (int r : {1, 4, 16, 64, 128, 256}) {
        float *d_a, *d_b;
        cudaMalloc(&d_a, r * HEAD_DIM * sizeof(float));
        cudaMalloc(&d_b, r * HEAD_DIM * sizeof(float));
        init_random_scaled<<<(r * HEAD_DIM + bl - 1) / bl, bl>>>(d_a, r * HEAD_DIM, 1.0f/sqrtf(HEAD_DIM), 300 + r);
        init_random_scaled<<<(r * HEAD_DIM + bl - 1) / bl, bl>>>(d_b, r * HEAD_DIM, 1.0f/sqrtf(r), 400 + r);
        cudaDeviceSynchronize();
        
        mlp_rank<<<gr, 32>>>(d_input, d_a, d_b, d_test_output, batch, r);
        compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_test_output, d_gt_output, d_errors, batch * HEAD_DIM);
        mse = sum_reduce(d_errors, batch * HEAD_DIM);
        rmse = sqrtf(mse);
        rel = rmse / output_mag * 100.0f;
        int params = r * HEAD_DIM * 2;
        printf("RANK-%-3d         | %7d | %.6f | %.4f   | %6.1f%%   | %s\n",
               r, params, mse, rmse, rel, rel < 20 ? "GOOD" : rel < 50 ? "OK" : "POOR");
        cudaFree(d_a); cudaFree(d_b);
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  Still RANDOM init - not trained. Next: actually FIT the shapes.\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    cudaFree(d_input); cudaFree(d_gt_output); cudaFree(d_test_output);
    cudaFree(d_errors); cudaFree(d_Wu); cudaFree(d_Wd);
    
    return 0;
}

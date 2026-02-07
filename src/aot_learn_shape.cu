// AOT SHAPE LEARNING
// Actually TRAIN the compressed shapes to match MLP output
// Simple gradient descent on the shape parameters

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

// Ground truth MLP
__global__ void mlp_forward(
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

// Low-rank forward: output = sum_r gelu(input · A_r) * B_r
__global__ void lowrank_forward(
    const float* __restrict__ input,
    const float* __restrict__ A,  // [rank, HEAD_DIM]
    const float* __restrict__ B,  // [rank, HEAD_DIM]
    float* __restrict__ output,
    int batch, int rank
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float in[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) in[d] = input[row * HEAD_DIM + d];
    
    float out[HEAD_DIM] = {0};
    for (int r = 0; r < rank; r++) {
        float proj = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) proj += in[d] * A[r * HEAD_DIM + d];
        // GELU
        proj = 0.5f * proj * (1.0f + tanhf(0.7978845608f * (proj + 0.044715f * proj * proj * proj)));
        for (int d = 0; d < HEAD_DIM; d++) out[d] += proj * B[r * HEAD_DIM + d];
    }
    
    for (int d = 0; d < HEAD_DIM; d++) output[row * HEAD_DIM + d] = out[d];
}

// Backward pass for low-rank: compute gradients for A and B
__global__ void lowrank_backward(
    const float* __restrict__ input,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ target,
    float* __restrict__ grad_A,  // [rank, HEAD_DIM]
    float* __restrict__ grad_B,  // [rank, HEAD_DIM]
    int batch, int rank
) {
    int r = blockIdx.x;  // one block per rank
    if (r >= rank) return;
    
    // Accumulate gradients across batch
    float local_grad_A[HEAD_DIM] = {0};
    float local_grad_B[HEAD_DIM] = {0};
    
    for (int row = threadIdx.x; row < batch; row += blockDim.x) {
        float in[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; d++) in[d] = input[row * HEAD_DIM + d];
        
        // Forward for this rank
        float proj = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) proj += in[d] * A[r * HEAD_DIM + d];
        
        // GELU and its derivative
        float x = proj;
        float tanh_arg = 0.7978845608f * (x + 0.044715f * x * x * x);
        float tanh_val = tanhf(tanh_arg);
        float gelu_out = 0.5f * x * (1.0f + tanh_val);
        float gelu_grad = 0.5f * (1.0f + tanh_val) + 
                          0.5f * x * (1.0f - tanh_val * tanh_val) * 
                          0.7978845608f * (1.0f + 0.134145f * x * x);
        
        // Compute full output to get error
        float out[HEAD_DIM] = {0};
        for (int rr = 0; rr < rank; rr++) {
            float p = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) p += in[d] * A[rr * HEAD_DIM + d];
            p = 0.5f * p * (1.0f + tanhf(0.7978845608f * (p + 0.044715f * p * p * p)));
            for (int d = 0; d < HEAD_DIM; d++) out[d] += p * B[rr * HEAD_DIM + d];
        }
        
        // Error = output - target
        float err[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; d++) err[d] = out[d] - target[row * HEAD_DIM + d];
        
        // grad_B[r,d] += err[d] * gelu_out
        // grad_A[r,d] += sum_d'(err[d'] * B[r,d']) * gelu_grad * in[d]
        float upstream = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) upstream += err[d] * B[r * HEAD_DIM + d];
        upstream *= gelu_grad;
        
        for (int d = 0; d < HEAD_DIM; d++) {
            local_grad_B[d] += err[d] * gelu_out;
            local_grad_A[d] += upstream * in[d];
        }
    }
    
    // Write gradients (simple - no atomic needed since one block per rank)
    for (int d = 0; d < HEAD_DIM; d++) {
        atomicAdd(&grad_A[r * HEAD_DIM + d], local_grad_A[d]);
        atomicAdd(&grad_B[r * HEAD_DIM + d], local_grad_B[d]);
    }
}

// SGD update
__global__ void sgd_update(float* params, const float* grads, int n, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        params[idx] -= lr * grads[idx];
    }
}

__global__ void zero_grads(float* grads, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grads[idx] = 0.0f;
}

__global__ void compute_mse(const float* pred, const float* target, float* errors, int n) {
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
    printf("  LEARNING COMPRESSED SHAPES\n");
    printf("  Train low-rank approximation to match MLP output\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    int batch = 1024;  // Training batch
    int bl = 256;
    
    // Allocate
    float *d_input, *d_target, *d_output, *d_errors;
    float *d_Wu, *d_Wd;
    cudaMalloc(&d_input, batch * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_target, batch * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_output, batch * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_errors, batch * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_Wu, HEAD_DIM * MLP_DIM * sizeof(float));
    cudaMalloc(&d_Wd, MLP_DIM * HEAD_DIM * sizeof(float));
    
    // Init MLP (the target we're trying to compress)
    float wu_scale = sqrtf(2.0f / HEAD_DIM);
    float wd_scale = sqrtf(2.0f / MLP_DIM);
    init_random_scaled<<<(HEAD_DIM * MLP_DIM + bl - 1) / bl, bl>>>(d_Wu, HEAD_DIM * MLP_DIM, wu_scale, 123);
    init_random_scaled<<<(MLP_DIM * HEAD_DIM + bl - 1) / bl, bl>>>(d_Wd, MLP_DIM * HEAD_DIM, wd_scale, 456);
    
    printf("Target MLP: %d parameters\n\n", HEAD_DIM * MLP_DIM + MLP_DIM * HEAD_DIM);
    
    // Test different ranks
    for (int rank : {4, 8, 16, 32, 64, 128}) {
        int num_params = rank * HEAD_DIM * 2;
        float compression = (float)(HEAD_DIM * MLP_DIM + MLP_DIM * HEAD_DIM) / num_params;
        
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("  RANK-%d: %d params (%.1fx compression)\n", rank, num_params, compression);
        printf("═══════════════════════════════════════════════════════════════════\n");
        
        // Allocate shape parameters
        float *d_A, *d_B, *d_grad_A, *d_grad_B;
        cudaMalloc(&d_A, rank * HEAD_DIM * sizeof(float));
        cudaMalloc(&d_B, rank * HEAD_DIM * sizeof(float));
        cudaMalloc(&d_grad_A, rank * HEAD_DIM * sizeof(float));
        cudaMalloc(&d_grad_B, rank * HEAD_DIM * sizeof(float));
        
        // Initialize
        init_random_scaled<<<(rank * HEAD_DIM + bl - 1) / bl, bl>>>(d_A, rank * HEAD_DIM, 0.1f, 1000 + rank);
        init_random_scaled<<<(rank * HEAD_DIM + bl - 1) / bl, bl>>>(d_B, rank * HEAD_DIM, 0.1f, 2000 + rank);
        cudaDeviceSynchronize();
        
        float lr = 0.001f;
        int gr = (batch + 31) / 32;
        
        printf("Step   | MSE        | RMSE    | Progress\n");
        printf("-------|------------|---------|----------\n");
        
        for (int step = 0; step <= 500; step++) {
            // Generate random input batch
            init_random_scaled<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_input, batch * HEAD_DIM, 2.0f, step * 1000);
            cudaDeviceSynchronize();
            
            // Get target output from real MLP
            mlp_forward<<<gr, 32>>>(d_input, d_Wu, d_Wd, d_target, batch);
            
            // Forward through our low-rank approximation
            lowrank_forward<<<gr, 32>>>(d_input, d_A, d_B, d_output, batch, rank);
            
            // Compute loss
            compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_output, d_target, d_errors, batch * HEAD_DIM);
            float mse = sum_reduce(d_errors, batch * HEAD_DIM);
            
            if (step % 50 == 0) {
                float rmse = sqrtf(mse);
                printf("%5d  | %.6f | %.5f | ", step, mse, rmse);
                int bars = (int)(50 * (1.0f - fminf(rmse / 0.1f, 1.0f)));
                for (int i = 0; i < bars; i++) printf("█");
                for (int i = bars; i < 50; i++) printf("░");
                printf("\n");
            }
            
            // Backward pass
            zero_grads<<<(rank * HEAD_DIM + bl - 1) / bl, bl>>>(d_grad_A, rank * HEAD_DIM);
            zero_grads<<<(rank * HEAD_DIM + bl - 1) / bl, bl>>>(d_grad_B, rank * HEAD_DIM);
            cudaDeviceSynchronize();
            
            lowrank_backward<<<rank, 128>>>(d_input, d_A, d_B, d_target, d_grad_A, d_grad_B, batch, rank);
            cudaDeviceSynchronize();
            
            // Update
            float scale_lr = lr / batch;
            sgd_update<<<(rank * HEAD_DIM + bl - 1) / bl, bl>>>(d_A, d_grad_A, rank * HEAD_DIM, scale_lr);
            sgd_update<<<(rank * HEAD_DIM + bl - 1) / bl, bl>>>(d_B, d_grad_B, rank * HEAD_DIM, scale_lr);
        }
        
        printf("\n");
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_grad_A); cudaFree(d_grad_B);
    }
    
    cudaFree(d_input); cudaFree(d_target); cudaFree(d_output); cudaFree(d_errors);
    cudaFree(d_Wu); cudaFree(d_Wd);
    
    return 0;
}

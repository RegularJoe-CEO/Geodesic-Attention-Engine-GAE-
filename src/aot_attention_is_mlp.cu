// ATTENTION IS THE MLP
// Instead of separate MLP, use attention over learned basis vectors
// The basis vectors ARE the MLP's knowledge, compressed

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

// NEW: Attention-as-MLP
// Input attends to N learned basis vectors (keys), retrieves their values
// This IS a mini-attention layer acting as the MLP
__global__ void attention_mlp(
    const float* __restrict__ input,      // [batch, HEAD_DIM] - acts as Q
    const float* __restrict__ basis_k,    // [num_basis, HEAD_DIM] - learned keys
    const float* __restrict__ basis_v,    // [num_basis, HEAD_DIM] - learned values
    float* __restrict__ output,
    int batch, int num_basis, float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = input[row * HEAD_DIM + d];
    
    // Compute attention over basis vectors (softmax)
    float max_score = -1e30f;
    float scores[512];  // max basis we'll test
    
    for (int b = 0; b < num_basis; b++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * basis_k[b * HEAD_DIM + d];
        s *= scale;
        scores[b] = s;
        max_score = fmaxf(max_score, s);
    }
    
    float sum_exp = 0.0f;
    for (int b = 0; b < num_basis; b++) {
        scores[b] = expf(scores[b] - max_score);
        sum_exp += scores[b];
    }
    
    // Weighted sum of basis values
    float out[HEAD_DIM] = {0};
    for (int b = 0; b < num_basis; b++) {
        float w = scores[b] / sum_exp;
        for (int d = 0; d < HEAD_DIM; d++) {
            out[d] += w * basis_v[b * HEAD_DIM + d];
        }
    }
    
    for (int d = 0; d < HEAD_DIM; d++) output[row * HEAD_DIM + d] = out[d];
}

// NEW: Gated attention-MLP (add learnable gate per basis)
__global__ void gated_attention_mlp(
    const float* __restrict__ input,
    const float* __restrict__ basis_k,
    const float* __restrict__ basis_v,
    const float* __restrict__ gates,      // [num_basis] - learned gates
    float* __restrict__ output,
    int batch, int num_basis, float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = input[row * HEAD_DIM + d];
    
    float max_score = -1e30f;
    float scores[512];
    
    for (int b = 0; b < num_basis; b++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * basis_k[b * HEAD_DIM + d];
        s *= scale;
        // Apply learned gate (like SwiGLU gating)
        s *= 1.0f / (1.0f + expf(-gates[b]));  // sigmoid gate
        scores[b] = s;
        max_score = fmaxf(max_score, s);
    }
    
    float sum_exp = 0.0f;
    for (int b = 0; b < num_basis; b++) {
        scores[b] = expf(scores[b] - max_score);
        sum_exp += scores[b];
    }
    
    float out[HEAD_DIM] = {0};
    for (int b = 0; b < num_basis; b++) {
        float w = scores[b] / sum_exp;
        for (int d = 0; d < HEAD_DIM; d++) {
            out[d] += w * basis_v[b * HEAD_DIM + d];
        }
    }
    
    for (int d = 0; d < HEAD_DIM; d++) output[row * HEAD_DIM + d] = out[d];
}

// NEW: Residual attention-MLP (input + attention_output)
__global__ void residual_attention_mlp(
    const float* __restrict__ input,
    const float* __restrict__ basis_k,
    const float* __restrict__ basis_v,
    float* __restrict__ output,
    int batch, int num_basis, float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = input[row * HEAD_DIM + d];
    
    float max_score = -1e30f;
    float scores[512];
    
    for (int b = 0; b < num_basis; b++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * basis_k[b * HEAD_DIM + d];
        s *= scale;
        scores[b] = s;
        max_score = fmaxf(max_score, s);
    }
    
    float sum_exp = 0.0f;
    for (int b = 0; b < num_basis; b++) {
        scores[b] = expf(scores[b] - max_score);
        sum_exp += scores[b];
    }
    
    for (int d = 0; d < HEAD_DIM; d++) {
        float att_out = 0.0f;
        for (int b = 0; b < num_basis; b++) {
            float w = scores[b] / sum_exp;
            att_out += w * basis_v[b * HEAD_DIM + d];
        }
        output[row * HEAD_DIM + d] = q[d] + att_out;  // residual
    }
}

__global__ void compute_mse(const float* pred, const float* target, float* errors, int n) {
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
    printf("  ATTENTION IS THE MLP\n");
    printf("  Replace MLP with attention over learned basis vectors\n");
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
    
    init_random_scaled<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_input, batch * HEAD_DIM, 2.0f, 42);
    float wu_scale = sqrtf(2.0f / HEAD_DIM);
    float wd_scale = sqrtf(2.0f / MLP_DIM);
    init_random_scaled<<<(HEAD_DIM * MLP_DIM + bl - 1) / bl, bl>>>(d_Wu, HEAD_DIM * MLP_DIM, wu_scale, 123);
    init_random_scaled<<<(MLP_DIM * HEAD_DIM + bl - 1) / bl, bl>>>(d_Wd, MLP_DIM * HEAD_DIM, wd_scale, 456);
    cudaDeviceSynchronize();
    
    int gr = (batch + 31) / 32;
    mlp_ground_truth<<<gr, 32>>>(d_input, d_Wu, d_Wd, d_gt_output, batch);
    cudaDeviceSynchronize();
    
    compute_magnitude<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_gt_output, d_errors, batch * HEAD_DIM);
    float output_mag = sqrtf(sum_reduce(d_errors, batch * HEAD_DIM));
    
    printf("Traditional MLP: 131,072 parameters\n");
    printf("GT Output RMS: %.4f\n\n", output_mag);
    
    printf("Approach              | Basis | Params  | RMSE   | vs Out | Quality\n");
    printf("----------------------|-------|---------|--------|--------|--------\n");
    
    float scale = 1.0f / sqrtf(HEAD_DIM);
    
    // Test different numbers of basis vectors
    for (int num_basis : {4, 8, 16, 32, 64, 128, 256, 512}) {
        size_t basis_size = num_basis * HEAD_DIM * sizeof(float);
        float *d_basis_k, *d_basis_v, *d_gates;
        cudaMalloc(&d_basis_k, basis_size);
        cudaMalloc(&d_basis_v, basis_size);
        cudaMalloc(&d_gates, num_basis * sizeof(float));
        
        // Initialize basis vectors
        init_random_scaled<<<(num_basis * HEAD_DIM + bl - 1) / bl, bl>>>(d_basis_k, num_basis * HEAD_DIM, wu_scale, 1000 + num_basis);
        init_random_scaled<<<(num_basis * HEAD_DIM + bl - 1) / bl, bl>>>(d_basis_v, num_basis * HEAD_DIM, wd_scale, 2000 + num_basis);
        init_random_scaled<<<(num_basis + bl - 1) / bl, bl>>>(d_gates, num_basis, 1.0f, 3000 + num_basis);
        cudaDeviceSynchronize();
        
        int params = num_basis * HEAD_DIM * 2;
        
        // Basic attention-MLP
        attention_mlp<<<gr, 32>>>(d_input, d_basis_k, d_basis_v, d_test_output, batch, num_basis, scale);
        compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_test_output, d_gt_output, d_errors, batch * HEAD_DIM);
        float rmse = sqrtf(sum_reduce(d_errors, batch * HEAD_DIM));
        float rel = rmse / output_mag * 100.0f;
        printf("ATTENTION-MLP         | %5d | %7d | %.4f | %5.1f%% | %s\n",
               num_basis, params, rmse, rel, rel < 50 ? "OK" : "POOR");
        
        // Gated attention-MLP
        gated_attention_mlp<<<gr, 32>>>(d_input, d_basis_k, d_basis_v, d_gates, d_test_output, batch, num_basis, scale);
        compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_test_output, d_gt_output, d_errors, batch * HEAD_DIM);
        rmse = sqrtf(sum_reduce(d_errors, batch * HEAD_DIM));
        rel = rmse / output_mag * 100.0f;
        printf("GGAED-ATT-MLP         | %5d | %7d | %.4f | %5.1f%% | %s\n",
               num_basis, params + num_basis, rmse, rel, rel < 50 ? "OK" : "POOR");
        
        // Residual attention-MLP
        residual_attention_mlp<<<gr, 32>>>(d_input, d_basis_k, d_basis_v, d_test_output, batch, num_basis, scale);
        compute_mse<<<(batch * HEAD_DIM + bl - 1) / bl, bl>>>(d_test_output, d_gt_output, d_errors, batch * HEAD_DIM);
        rmse = sqrtf(sum_reduce(d_errors, batch * HEAD_DIM));
        rel = rmse / output_mag * 100.0f;
        printf("RESIDUAL-ATT-MLP      | %5d | %7d | %.4f | %5.1f%% | %s\n",
               num_basis, params, rmse, rel, rel < 50 ? "OK" : "POOR");
        
        printf("----------------------|-------|---------|--------|--------|--------\n");
        
        cudaFree(d_basis_k); cudaFree(d_basis_v); cudaFree(d_gates);
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  KEY INSIGHT: If we can match MLP with ~32-64 basis vectors,\n");
    printf("  that's 8K-16K params vs 131K = 8-16x compression\n");
    printf("  AND it's just attention - no separate MLP kernel needed!\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    cudaFree(d_input); cudaFree(d_gt_output); cudaFree(d_test_output);
    cudaFree(d_errors); cudaFree(d_Wu); cudaFree(d_Wd);
    
    return 0;
}

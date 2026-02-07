// AOT DIRECT SHAPE ENGINE
// No DCT transform/inverse - the shape IS the operation
// Exploring the minimal representation that can replace MLP

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HEAD_DIM 128

// APPROACH 1: Minimal - just scale + bias + nonlinearity
// 256 parameters total
__global__ void attention_minimal_mlp(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ scale,  // [HEAD_DIM]
    const float* __restrict__ bias,   // [HEAD_DIM]
    float* __restrict__ output,
    int seq_len, float qk_scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = Q[row * HEAD_DIM + d];
    
    float rmax = -1e30f, rsum = 0.0f;
    float att[HEAD_DIM] = {0};
    
    for (int c = 0; c <= row; c++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * K[c * HEAD_DIM + d];
        s *= qk_scale;
        float om = rmax; rmax = fmaxf(rmax, s);
        float rs = expf(om - rmax);
        rsum = rsum * rs + expf(s - rmax);
        float w = expf(s - rmax);
        for (int d = 0; d < HEAD_DIM; d++) att[d] = att[d] * rs + w * V[c * HEAD_DIM + d];
    }
    float inv = 1.0f / rsum;
    for (int d = 0; d < HEAD_DIM; d++) att[d] *= inv;
    
    // MINIMAL MLP: scale, shift, gelu
    for (int d = 0; d < HEAD_DIM; d++) {
        float x = att[d] * scale[d] + bias[d];
        float g = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[row * HEAD_DIM + d] = att[d] + g;
    }
}

// APPROACH 2: Small learned kernel convolution
// Like a 1D conv with small kernel across dimensions
__global__ void attention_conv_mlp(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ kernel,  // [kernel_size] - shared across all dims
    float* __restrict__ output,
    int seq_len, int kernel_size, float qk_scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = Q[row * HEAD_DIM + d];
    
    float rmax = -1e30f, rsum = 0.0f;
    float att[HEAD_DIM] = {0};
    
    for (int c = 0; c <= row; c++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * K[c * HEAD_DIM + d];
        s *= qk_scale;
        float om = rmax; rmax = fmaxf(rmax, s);
        float rs = expf(om - rmax);
        rsum = rsum * rs + expf(s - rmax);
        float w = expf(s - rmax);
        for (int d = 0; d < HEAD_DIM; d++) att[d] = att[d] * rs + w * V[c * HEAD_DIM + d];
    }
    float inv = 1.0f / rsum;
    for (int d = 0; d < HEAD_DIM; d++) att[d] *= inv;
    
    // CONV MLP: convolve across dimension with small kernel
    int half_k = kernel_size / 2;
    float conv[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        conv[d] = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int src = d - half_k + k;
            if (src >= 0 && src < HEAD_DIM) {
                conv[d] += att[src] * kernel[k];
            }
        }
        // GELU
        float x = conv[d];
        conv[d] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
    
    for (int d = 0; d < HEAD_DIM; d++) output[row * HEAD_DIM + d] = att[d] + conv[d];
}

// APPROACH 3: Outer product shape - rank-1 approximation of full MLP
// shape_a[HEAD_DIM] ⊗ shape_b[HEAD_DIM] approximates Wu @ Wd
__global__ void attention_rank1_mlp(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ shape_a,  // [HEAD_DIM]
    const float* __restrict__ shape_b,  // [HEAD_DIM]
    float* __restrict__ output,
    int seq_len, float qk_scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = Q[row * HEAD_DIM + d];
    
    float rmax = -1e30f, rsum = 0.0f;
    float att[HEAD_DIM] = {0};
    
    for (int c = 0; c <= row; c++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * K[c * HEAD_DIM + d];
        s *= qk_scale;
        float om = rmax; rmax = fmaxf(rmax, s);
        float rs = expf(om - rmax);
        rsum = rsum * rs + expf(s - rmax);
        float w = expf(s - rmax);
        for (int d = 0; d < HEAD_DIM; d++) att[d] = att[d] * rs + w * V[c * HEAD_DIM + d];
    }
    float inv = 1.0f / rsum;
    for (int d = 0; d < HEAD_DIM; d++) att[d] *= inv;
    
    // RANK-1 MLP: (att · shape_a) * shape_b
    // This is: for each output dim, contribution is weighted by dot product
    float projection = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) projection += att[d] * shape_a[d];
    
    // GELU on the scalar projection
    projection = 0.5f * projection * (1.0f + tanhf(0.7978845608f * (projection + 0.044715f * projection * projection * projection)));
    
    for (int d = 0; d < HEAD_DIM; d++) {
        output[row * HEAD_DIM + d] = att[d] + projection * shape_b[d];
    }
}

// APPROACH 4: Low-rank (rank-R) approximation
__global__ void attention_rankR_mlp(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ shapes_a,  // [R, HEAD_DIM]
    const float* __restrict__ shapes_b,  // [R, HEAD_DIM]
    float* __restrict__ output,
    int seq_len, int rank, float qk_scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = Q[row * HEAD_DIM + d];
    
    float rmax = -1e30f, rsum = 0.0f;
    float att[HEAD_DIM] = {0};
    
    for (int c = 0; c <= row; c++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * K[c * HEAD_DIM + d];
        s *= qk_scale;
        float om = rmax; rmax = fmaxf(rmax, s);
        float rs = expf(om - rmax);
        rsum = rsum * rs + expf(s - rmax);
        float w = expf(s - rmax);
        for (int d = 0; d < HEAD_DIM; d++) att[d] = att[d] * rs + w * V[c * HEAD_DIM + d];
    }
    float inv = 1.0f / rsum;
    for (int d = 0; d < HEAD_DIM; d++) att[d] *= inv;
    
    // RANK-R MLP: sum of R rank-1 matrices
    float mlp_out[HEAD_DIM] = {0};
    for (int r = 0; r < rank; r++) {
        float proj = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) proj += att[d] * shapes_a[r * HEAD_DIM + d];
        proj = 0.5f * proj * (1.0f + tanhf(0.7978845608f * (proj + 0.044715f * proj * proj * proj)));
        for (int d = 0; d < HEAD_DIM; d++) mlp_out[d] += proj * shapes_b[r * HEAD_DIM + d];
    }
    
    for (int d = 0; d < HEAD_DIM; d++) output[row * HEAD_DIM + d] = att[d] + mlp_out[d];
}

// Traditional for comparison
__global__ void traditional_layer(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ Wu, const float* __restrict__ Wd,
    float* __restrict__ output, int seq_len, int mlp_dim, float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = Q[row * HEAD_DIM + d];
    
    float rmax = -1e30f, rsum = 0.0f;
    float att[HEAD_DIM] = {0};
    
    for (int c = 0; c <= row; c++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * K[c * HEAD_DIM + d];
        s *= scale;
        float om = rmax; rmax = fmaxf(rmax, s);
        float rs = expf(om - rmax);
        rsum = rsum * rs + expf(s - rmax);
        float w = expf(s - rmax);
        for (int d = 0; d < HEAD_DIM; d++) att[d] = att[d] * rs + w * V[c * HEAD_DIM + d];
    }
    float inv = 1.0f / rsum;
    for (int d = 0; d < HEAD_DIM; d++) att[d] *= inv;
    
    float mlp[HEAD_DIM] = {0};
    for (int m = 0; m < mlp_dim; m++) {
        float h = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) h += att[d] * Wu[d * mlp_dim + m];
        h = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
        for (int d = 0; d < HEAD_DIM; d++) mlp[d] += h * Wd[m * HEAD_DIM + d];
    }
    
    for (int d = 0; d < HEAD_DIM; d++) output[row * HEAD_DIM + d] = att[d] + mlp[d];
}

float bench_minimal(int seq) {
    size_t us = seq * HEAD_DIM * sizeof(float);
    size_t ps = HEAD_DIM * sizeof(float);
    
    float *dQ, *dK, *dV, *dO, *d_scale, *d_bias;
    cudaMalloc(&dQ, us); cudaMalloc(&dK, us); cudaMalloc(&dV, us);
    cudaMalloc(&dO, us); cudaMalloc(&d_scale, ps); cudaMalloc(&d_bias, ps);
    
    float sc = 1.0f / sqrtf(HEAD_DIM);
    int bl = 32, gr = (seq + bl - 1) / bl;
    
    for (int i = 0; i < 2; i++) attention_minimal_mlp<<<gr, bl>>>(dQ, dK, dV, d_scale, d_bias, dO, seq, sc);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    attention_minimal_mlp<<<gr, bl>>>(dQ, dK, dV, d_scale, d_bias, dO, seq, sc);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(d_scale); cudaFree(d_bias);
    return ms;
}

float bench_conv(int seq, int kernel_size) {
    size_t us = seq * HEAD_DIM * sizeof(float);
    size_t ks = kernel_size * sizeof(float);
    
    float *dQ, *dK, *dV, *dO, *d_kernel;
    cudaMalloc(&dQ, us); cudaMalloc(&dK, us); cudaMalloc(&dV, us);
    cudaMalloc(&dO, us); cudaMalloc(&d_kernel, ks);
    
    float sc = 1.0f / sqrtf(HEAD_DIM);
    int bl = 32, gr = (seq + bl - 1) / bl;
    
    for (int i = 0; i < 2; i++) attention_conv_mlp<<<gr, bl>>>(dQ, dK, dV, d_kernel, dO, seq, kernel_size, sc);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    attention_conv_mlp<<<gr, bl>>>(dQ, dK, dV, d_kernel, dO, seq, kernel_size, sc);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(d_kernel);
    return ms;
}

float bench_rank(int seq, int rank) {
    size_t us = seq * HEAD_DIM * sizeof(float);
    size_t rs = rank * HEAD_DIM * sizeof(float);
    
    float *dQ, *dK, *dV, *dO, *d_a, *d_b;
    cudaMalloc(&dQ, us); cudaMalloc(&dK, us); cudaMalloc(&dV, us);
    cudaMalloc(&dO, us); cudaMalloc(&d_a, rs); cudaMalloc(&d_b, rs);
    
    float sc = 1.0f / sqrtf(HEAD_DIM);
    int bl = 32, gr = (seq + bl - 1) / bl;
    
    for (int i = 0; i < 2; i++) attention_rankR_mlp<<<gr, bl>>>(dQ, dK, dV, d_a, d_b, dO, seq, rank, sc);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    attention_rankR_mlp<<<gr, bl>>>(dQ, dK, dV, d_a, d_b, dO, seq, rank, sc);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(d_a); cudaFree(d_b);
    return ms;
}

float bench_trad(int seq) {
    int md = 4 * HEAD_DIM;
    size_t us = seq * HEAD_DIM * sizeof(float);
    size_t ws = HEAD_DIM * md * sizeof(float);
    
    float *dQ, *dK, *dV, *dWu, *dWd, *dO;
    cudaMalloc(&dQ, us); cudaMalloc(&dK, us); cudaMalloc(&dV, us);
    cudaMalloc(&dWu, ws); cudaMalloc(&dWd, ws); cudaMalloc(&dO, us);
    
    float sc = 1.0f / sqrtf(HEAD_DIM);
    int bl = 32, gr = (seq + bl - 1) / bl;
    
    for (int i = 0; i < 2; i++) traditional_layer<<<gr, bl>>>(dQ, dK, dV, dWu, dWd, dO, seq, md, sc);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    traditional_layer<<<gr, bl>>>(dQ, dK, dV, dWu, dWd, dO, seq, md, sc);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dWu); cudaFree(dWd); cudaFree(dO);
    return ms;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  DIRECT SHAPE MLP: No transforms, just the shape itself\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("Traditional MLP: 131,072 parameters\n\n");
    
    for (int seq : {65536, 262144}) {
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("  @ %d tokens\n", seq);
        printf("═══════════════════════════════════════════════════════════════════\n\n");
        
        float trad = bench_trad(seq);
        printf("TRADITIONAL: %.1f ms\n\n", trad);
        
        printf("Approach         | Params  | Time (ms) | vs TRAD  | Compression\n");
        printf("-----------------|---------|-----------|----------|------------\n");
        
        // Minimal
        float min_ms = bench_minimal(seq);
        printf("MINIMAL (s+b)    | %7d | %9.1f | %+6.1f%% | %.0fx\n", 
               256, min_ms, (min_ms-trad)/trad*100, 131072.0f/256);
        
        // Conv kernels
        for (int k : {3, 5, 7, 9}) {
            float c_ms = bench_conv(seq, k);
            printf("CONV kernel=%d    | %7d | %9.1f | %+6.1f%% | %.0fx\n",
                   k, k, c_ms, (c_ms-trad)/trad*100, 131072.0f/k);
        }
        
        // Rank approximations
        for (int r : {1, 2, 4, 8, 16}) {
            float r_ms = bench_rank(seq, r);
            int params = r * HEAD_DIM * 2;
            printf("RANK-%d           | %7d | %9.1f | %+6.1f%% | %.0fx\n",
                   r, params, r_ms, (r_ms-trad)/trad*100, 131072.0f/params);
        }
        
        printf("\n");
    }
    
    return 0;
}

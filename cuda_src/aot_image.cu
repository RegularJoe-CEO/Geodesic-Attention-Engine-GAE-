// AOT IMAGE ENGINE
// The MLP transformation as a compressed image
// Knowledge tokens become basis functions (like DCT/Fourier)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HEAD_DIM 128
#define PI 3.14159265358979f

// Precompute DCT basis functions - these ARE the "knowledge tokens"
// But they're not learned, they're mathematical basis functions
__constant__ float DCT_BASIS[128 * 128];  // 128 basis functions of dim 128

__global__ void precompute_dct_basis(float* basis, int N) {
    int k = blockIdx.x;  // basis index
    int n = threadIdx.x; // dimension index
    
    if (k < N && n < N) {
        float scale = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
        basis[k * N + n] = scale * cosf(PI * k * (2.0f * n + 1.0f) / (2.0f * N));
    }
}

// The MLP "image" - coefficients in DCT space
// Instead of 128x512 weights, we have 128 DCT coefficients per output dim
// This is the compressed representation

__global__ void attention_with_image_mlp(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mlp_image,  // [HEAD_DIM, num_coeffs] - compressed MLP
    const float* __restrict__ dct_basis,  // [num_coeffs, HEAD_DIM] - basis functions
    float* __restrict__ output,
    int seq_len, int num_coeffs, float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = Q[row * HEAD_DIM + d];
    
    // Standard causal attention
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
    
    // IMAGE-BASED MLP
    // Step 1: Transform attention output to DCT space (compress)
    float dct_coeffs[128];  // max coefficients we'd use
    for (int k = 0; k < num_coeffs; k++) {
        dct_coeffs[k] = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            dct_coeffs[k] += att[d] * dct_basis[k * HEAD_DIM + d];
        }
    }
    
    // Step 2: Apply the "image" - element-wise multiply in frequency space
    // This is like applying a filter to an image!
    for (int k = 0; k < num_coeffs; k++) {
        dct_coeffs[k] *= mlp_image[k];  // The learned "image" 
    }
    
    // Step 3: Inverse DCT (reconstruct)
    float mlp_out[HEAD_DIM] = {0};
    for (int d = 0; d < HEAD_DIM; d++) {
        for (int k = 0; k < num_coeffs; k++) {
            mlp_out[d] += dct_coeffs[k] * dct_basis[k * HEAD_DIM + d];
        }
    }
    
    // Residual connection
    for (int d = 0; d < HEAD_DIM; d++) {
        output[row * HEAD_DIM + d] = att[d] + mlp_out[d];
    }
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

float bench_image(int seq, int num_coeffs) {
    size_t us = seq * HEAD_DIM * sizeof(float);
    size_t basis_size = HEAD_DIM * HEAD_DIM * sizeof(float);
    size_t image_size = num_coeffs * sizeof(float);
    
    float *dQ, *dK, *dV, *dO, *d_basis, *d_image;
    cudaMalloc(&dQ, us); cudaMalloc(&dK, us); cudaMalloc(&dV, us);
    cudaMalloc(&dO, us); cudaMalloc(&d_basis, basis_size); cudaMalloc(&d_image, image_size);
    
    // Initialize DCT basis
    precompute_dct_basis<<<HEAD_DIM, HEAD_DIM>>>(d_basis, HEAD_DIM);
    
    // Random MLP image (in real use, this would be learned)
    float *h_image = (float*)malloc(image_size);
    for (int i = 0; i < num_coeffs; i++) h_image[i] = 1.0f / (i + 1);  // decay
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    free(h_image);
    
    float sc = 1.0f / sqrtf(HEAD_DIM);
    int bl = 32, gr = (seq + bl - 1) / bl;
    
    // Warmup
    for (int i = 0; i < 2; i++) {
        attention_with_image_mlp<<<gr, bl>>>(dQ, dK, dV, d_image, d_basis, dO, seq, num_coeffs, sc);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    attention_with_image_mlp<<<gr, bl>>>(dQ, dK, dV, d_image, d_basis, dO, seq, num_coeffs, sc);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(d_basis); cudaFree(d_image);
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
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    traditional_layer<<<gr, bl>>>(dQ, dK, dV, dWu, dWd, dO, seq, md, sc);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dWu); cudaFree(dWd); cudaFree(dO);
    return ms;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  IMAGE-BASED MLP: The transformation as a compressed picture\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("Traditional MLP: 128 × 512 × 2 = 131,072 parameters\n");
    printf("Image MLP: Just N DCT coefficients\n\n");
    
    int seq = 65536;
    float trad = bench_trad(seq);
    printf("TRADITIONAL @ %d: %.1f ms\n\n", seq, trad);
    
    printf("DCT Coeffs | Params | IMAGE MLP (ms) | vs TRAD  | Compression\n");
    printf("-----------|--------|----------------|----------|------------\n");
    
    int coeffs[] = {8, 16, 32, 64, 128};
    for (int i = 0; i < 5; i++) {
        int nc = coeffs[i];
        float img_ms = bench_image(seq, nc);
        float diff = (img_ms - trad) / trad * 100.0f;
        float compression = 131072.0f / nc;
        printf("%10d | %6d | %12.1f   | %+6.1f%%  | %.0fx\n", 
               nc, nc, img_ms, diff, compression);
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  @ 262144 tokens\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    seq = 262144;
    trad = bench_trad(seq);
    printf("TRADITIONAL @ %d: %.1f ms\n\n", seq, trad);
    
    printf("DCT Coeffs | IMAGE MLP (ms) | vs TRAD\n");
    printf("-----------|----------------|--------\n");
    
    for (int i = 0; i < 5; i++) {
        int nc = coeffs[i];
        float img_ms = bench_image(seq, nc);
        float diff = (img_ms - trad) / trad * 100.0f;
        printf("%10d | %12.1f   | %+.1f%%\n", nc, img_ms, diff);
    }
    
    return 0;
}

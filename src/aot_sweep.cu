#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HEAD_DIM 128

__global__ void attention_only(
    const float* __restrict__ Q,
    const float* __restrict__ user_K,
    const float* __restrict__ user_V,
    const float* __restrict__ know_K,
    const float* __restrict__ know_V,
    float* __restrict__ output,
    int seq_len, int num_know, float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) q[d] = Q[row * HEAD_DIM + d];
    
    float rmax = -1e30f, rsum = 0.0f;
    float out[HEAD_DIM] = {0};
    
    for (int k = 0; k < num_know; k++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * know_K[k * HEAD_DIM + d];
        s *= scale;
        float om = rmax; rmax = fmaxf(rmax, s);
        float rs = expf(om - rmax);
        rsum = rsum * rs + expf(s - rmax);
        float w = expf(s - rmax);
        for (int d = 0; d < HEAD_DIM; d++) out[d] = out[d] * rs + w * know_V[k * HEAD_DIM + d];
    }
    
    for (int c = 0; c <= row; c++) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * user_K[c * HEAD_DIM + d];
        s *= scale;
        float om = rmax; rmax = fmaxf(rmax, s);
        float rs = expf(om - rmax);
        rsum = rsum * rs + expf(s - rmax);
        float w = expf(s - rmax);
        for (int d = 0; d < HEAD_DIM; d++) out[d] = out[d] * rs + w * user_V[c * HEAD_DIM + d];
    }
    
    float inv = 1.0f / rsum;
    for (int d = 0; d < HEAD_DIM; d++) output[row * HEAD_DIM + d] = out[d] * inv;
}

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

float bench_aot(int seq, int nk) {
    size_t us = seq * HEAD_DIM * sizeof(float);
    size_t ks = nk * HEAD_DIM * sizeof(float);
    
    float *dQ, *duK, *duV, *dkK, *dkV, *dO;
    cudaMalloc(&dQ, us); cudaMalloc(&duK, us); cudaMalloc(&duV, us);
    cudaMalloc(&dkK, ks); cudaMalloc(&dkV, ks); cudaMalloc(&dO, us);
    
    float sc = 1.0f / sqrtf(HEAD_DIM);
    int bl = 32, gr = (seq + bl - 1) / bl;
    
    for (int i = 0; i < 2; i++) attention_only<<<gr, bl>>>(dQ, duK, duV, dkK, dkV, dO, seq, nk, sc);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    attention_only<<<gr, bl>>>(dQ, duK, duV, dkK, dkV, dO, seq, nk, sc);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    
    cudaFree(dQ); cudaFree(duK); cudaFree(duV); cudaFree(dkK); cudaFree(dkV); cudaFree(dO);
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
    printf("  AOT KNOWLEDGE TOKEN SWEEP @ 65536 (crossover point)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    int seq = 65536;
    float trad = bench_trad(seq);
    printf("TRAD (Attn+MLP): %.1f ms\n\n", trad);
    
    printf("Knowledge Tokens | AOT Time  | vs TRAD\n");
    printf("-----------------|-----------|--------\n");
    
    int know_counts[] = {64, 128, 256, 512, 1024, 2048, 4096};
    for (int i = 0; i < 7; i++) {
        float aot = bench_aot(seq, know_counts[i]);
        float diff = (aot - trad) / trad * 100.0f;
        printf("%16d | %7.1f ms | %+.1f%%\n", know_counts[i], aot, diff);
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  AOT @ 262144 (GAE territory) - varying knowledge tokens\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    seq = 262144;
    trad = bench_trad(seq);
    printf("TRAD (Attn+MLP): %.1f ms\n\n", trad);
    
    printf("Knowledge Tokens | AOT Time  | vs TRAD\n");
    printf("-----------------|-----------|--------\n");
    
    for (int i = 0; i < 7; i++) {
        float aot = bench_aot(seq, know_counts[i]);
        float diff = (aot - trad) / trad * 100.0f;
        printf("%16d | %7.1f ms | %+.1f%%\n", know_counts[i], aot, diff);
    }
    
    return 0;
}

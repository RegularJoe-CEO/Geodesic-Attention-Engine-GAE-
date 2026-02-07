// AOT ENGINE - Attention-Only Transformer
// No MLP. Knowledge tokens replace computation.

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
    
    // Attend to knowledge (replaces MLP)
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
    
    // Attend to user tokens (causal)
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
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ Wu,
    const float* __restrict__ Wd,
    float* __restrict__ output,
    int seq_len, int mlp_dim, float scale
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
    
    // MLP
    float mlp[HEAD_DIM] = {0};
    for (int m = 0; m < mlp_dim; m++) {
        float h = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) h += att[d] * Wu[d * mlp_dim + m];
        h = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
        for (int d = 0; d < HEAD_DIM; d++) mlp[d] += h * Wd[m * HEAD_DIM + d];
    }
    
    for (int d = 0; d < HEAD_DIM; d++) output[row * HEAD_DIM + d] = att[d] + mlp[d];
}

void bench_aot(int seq, int nk, int iters) {
    printf("\nAOT (Attn-Only): Seq %d | Know %d\n", seq, nk);
    
    size_t us = seq * HEAD_DIM * sizeof(float);
    size_t ks = nk * HEAD_DIM * sizeof(float);
    
    float *hQ, *huK, *huV, *hkK, *hkV;
    hQ = (float*)malloc(us); huK = (float*)malloc(us); huV = (float*)malloc(us);
    hkK = (float*)malloc(ks); hkV = (float*)malloc(ks);
    
    for (int i = 0; i < seq * HEAD_DIM; i++) { hQ[i] = 0.01f; huK[i] = 0.01f; huV[i] = 0.01f; }
    for (int i = 0; i < nk * HEAD_DIM; i++) { hkK[i] = 0.01f; hkV[i] = 0.01f; }
    
    float *dQ, *duK, *duV, *dkK, *dkV, *dO;
    cudaMalloc(&dQ, us); cudaMalloc(&duK, us); cudaMalloc(&duV, us);
    cudaMalloc(&dkK, ks); cudaMalloc(&dkV, ks); cudaMalloc(&dO, us);
    
    cudaMemcpy(dQ, hQ, us, cudaMemcpyHostToDevice);
    cudaMemcpy(duK, huK, us, cudaMemcpyHostToDevice);
    cudaMemcpy(duV, huV, us, cudaMemcpyHostToDevice);
    cudaMemcpy(dkK, hkK, ks, cudaMemcpyHostToDevice);
    cudaMemcpy(dkV, hkV, ks, cudaMemcpyHostToDevice);
    
    float sc = 1.0f / sqrtf(HEAD_DIM);
    int bl = 32, gr = (seq + bl - 1) / bl;
    
    for (int i = 0; i < 3; i++) attention_only<<<gr, bl>>>(dQ, duK, duV, dkK, dkV, dO, seq, nk, sc);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) attention_only<<<gr, bl>>>(dQ, duK, duV, dkK, dkV, dO, seq, nk, sc);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    ms /= iters;
    
    double kf = (double)seq * nk * HEAD_DIM * 2.0;
    double uf = (double)seq * (seq + 1.0) / 2.0 * HEAD_DIM * 2.0;
    double tf = (kf + uf) / (ms / 1000.0) / 1e12;
    
    printf("Time: %.3f ms | %.3f TFLOPS\n", ms, tf);
    
    cudaFree(dQ); cudaFree(duK); cudaFree(duV); cudaFree(dkK); cudaFree(dkV); cudaFree(dO);
    free(hQ); free(huK); free(huV); free(hkK); free(hkV);
}

void bench_trad(int seq, int iters) {
    printf("\nTRAD (Attn+MLP): Seq %d | MLP %d\n", seq, 4*HEAD_DIM);
    
    int md = 4 * HEAD_DIM;
    size_t us = seq * HEAD_DIM * sizeof(float);
    size_t ws = HEAD_DIM * md * sizeof(float);
    
    float *hQ, *hK, *hV, *hWu, *hWd;
    hQ = (float*)malloc(us); hK = (float*)malloc(us); hV = (float*)malloc(us);
    hWu = (float*)malloc(ws); hWd = (float*)malloc(ws);
    
    for (int i = 0; i < seq * HEAD_DIM; i++) { hQ[i] = 0.01f; hK[i] = 0.01f; hV[i] = 0.01f; }
    for (size_t i = 0; i < ws/sizeof(float); i++) { hWu[i] = 0.01f; hWd[i] = 0.01f; }
    
    float *dQ, *dK, *dV, *dWu, *dWd, *dO;
    cudaMalloc(&dQ, us); cudaMalloc(&dK, us); cudaMalloc(&dV, us);
    cudaMalloc(&dWu, ws); cudaMalloc(&dWd, ws); cudaMalloc(&dO, us);
    
    cudaMemcpy(dQ, hQ, us, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, us, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, us, cudaMemcpyHostToDevice);
    cudaMemcpy(dWu, hWu, ws, cudaMemcpyHostToDevice);
    cudaMemcpy(dWd, hWd, ws, cudaMemcpyHostToDevice);
    
    float sc = 1.0f / sqrtf(HEAD_DIM);
    int bl = 32, gr = (seq + bl - 1) / bl;
    
    for (int i = 0; i < 3; i++) traditional_layer<<<gr, bl>>>(dQ, dK, dV, dWu, dWd, dO, seq, md, sc);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) traditional_layer<<<gr, bl>>>(dQ, dK, dV, dWu, dWd, dO, seq, md, sc);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    ms /= iters;
    
    double af = (double)seq * (seq + 1.0) / 2.0 * HEAD_DIM * 2.0;
    double mf = (double)seq * HEAD_DIM * md * 4.0;
    double tf = (af + mf) / (ms / 1000.0) / 1e12;
    
    printf("Time: %.3f ms | %.3f TFLOPS | Attn %.0f%% MLP %.0f%%\n", ms, tf, 100*af/(af+mf), 100*mf/(af+mf));
    
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dWu); cudaFree(dWd); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hWu); free(hWd);
}

int main() {
    printf("════════════════════════════════════════════════════════════\n");
    printf("  AOT ENGINE - Attention-Only Transformer\n");
    printf("  Knowledge tokens replace MLP computation\n");
    printf("════════════════════════════════════════════════════════════\n");
    
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s\n", p.name);
    
    int seqs[] = {8192, 16384, 32768, 65536, 131072};
    for (int i = 0; i < 5; i++) {
        int s = seqs[i];
        int it = s <= 32768 ? 10 : 3;
        bench_trad(s, it);
        bench_aot(s, 1024, it);
        printf("────────────────────────────────────────────────────────────\n");
    }
    return 0;
}

// AOT ENGINE - LOOP FREE VERSION
// Knowledge tokens processed via warp-parallel reduction
// No sequential loop over knowledge tokens

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HEAD_DIM 128
#define WARP_SIZE 32

// Warp-parallel knowledge attention - each thread handles subset of knowledge tokens
__global__ void attention_only_loopfree(
    const float* __restrict__ Q,
    const float* __restrict__ user_K,
    const float* __restrict__ user_V,
    const float* __restrict__ know_K,  // [num_know, HEAD_DIM]
    const float* __restrict__ know_V,
    float* __restrict__ output,
    int seq_len, int num_know, float scale
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;  // 0-31 within warp
    
    if (row >= seq_len) return;
    
    // Load query into registers (all threads load same Q)
    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        q[d] = Q[row * HEAD_DIM + d];
    }
    
    // Each thread computes partial online softmax over its knowledge tokens
    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_out[HEAD_DIM] = {0};
    
    // Thread i handles knowledge tokens: i, i+32, i+64, ...
    for (int k = tid; k < num_know; k += WARP_SIZE) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            s += q[d] * know_K[k * HEAD_DIM + d];
        }
        s *= scale;
        
        float om = local_max;
        local_max = fmaxf(local_max, s);
        float rs = expf(om - local_max);
        local_sum = local_sum * rs + expf(s - local_max);
        float w = expf(s - local_max);
        for (int d = 0; d < HEAD_DIM; d++) {
            local_out[d] = local_out[d] * rs + w * know_V[k * HEAD_DIM + d];
        }
    }
    
    // Warp reduce: merge all threads' online softmax states
    // This is the magic - combine partial softmax results
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
        
        float new_max = fmaxf(local_max, other_max);
        float scale_self = expf(local_max - new_max);
        float scale_other = expf(other_max - new_max);
        
        local_sum = local_sum * scale_self + other_sum * scale_other;
        
        for (int d = 0; d < HEAD_DIM; d++) {
            float other_out = __shfl_down_sync(0xffffffff, local_out[d], offset);
            local_out[d] = local_out[d] * scale_self + other_out * scale_other;
        }
        
        local_max = new_max;
    }
    
    // Thread 0 has merged knowledge attention, now do user attention
    if (tid == 0) {
        float rmax = local_max;
        float rsum = local_sum;
        float out[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; d++) out[d] = local_out[d];
        
        // User tokens (causal) - sequential but that's O(N) not O(K)
        for (int c = 0; c <= row; c++) {
            float s = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                s += q[d] * user_K[c * HEAD_DIM + d];
            }
            s *= scale;
            
            float om = rmax;
            rmax = fmaxf(rmax, s);
            float rs = expf(om - rmax);
            rsum = rsum * rs + expf(s - rmax);
            float w = expf(s - rmax);
            for (int d = 0; d < HEAD_DIM; d++) {
                out[d] = out[d] * rs + w * user_V[c * HEAD_DIM + d];
            }
        }
        
        float inv = 1.0f / rsum;
        for (int d = 0; d < HEAD_DIM; d++) {
            output[row * HEAD_DIM + d] = out[d] * inv;
        }
    }
}

// Original loop version for comparison
__global__ void attention_only_loop(
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

float bench(int seq, int nk, bool loopfree) {
    size_t us = seq * HEAD_DIM * sizeof(float);
    size_t ks = nk * HEAD_DIM * sizeof(float);
    
    float *dQ, *duK, *duV, *dkK, *dkV, *dO;
    cudaMalloc(&dQ, us); cudaMalloc(&duK, us); cudaMalloc(&duV, us);
    cudaMalloc(&dkK, ks); cudaMalloc(&dkV, ks); cudaMalloc(&dO, us);
    
    // Initialize
    float *h = (float*)malloc(us > ks ? us : ks);
    for (int i = 0; i < (us > ks ? us : ks)/sizeof(float); i++) h[i] = 0.01f;
    cudaMemcpy(dQ, h, us, cudaMemcpyHostToDevice);
    cudaMemcpy(duK, h, us, cudaMemcpyHostToDevice);
    cudaMemcpy(duV, h, us, cudaMemcpyHostToDevice);
    cudaMemcpy(dkK, h, ks, cudaMemcpyHostToDevice);
    cudaMemcpy(dkV, h, ks, cudaMemcpyHostToDevice);
    free(h);
    
    float sc = 1.0f / sqrtf(HEAD_DIM);
    
    // Warmup
    for (int i = 0; i < 2; i++) {
        if (loopfree) {
            attention_only_loopfree<<<seq, WARP_SIZE>>>(dQ, duK, duV, dkK, dkV, dO, seq, nk, sc);
        } else {
            int bl = 32, gr = (seq + bl - 1) / bl;
            attention_only_loop<<<gr, bl>>>(dQ, duK, duV, dkK, dkV, dO, seq, nk, sc);
        }
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    
    if (loopfree) {
        attention_only_loopfree<<<seq, WARP_SIZE>>>(dQ, duK, duV, dkK, dkV, dO, seq, nk, sc);
    } else {
        int bl = 32, gr = (seq + bl - 1) / bl;
        attention_only_loop<<<gr, bl>>>(dQ, duK, duV, dkK, dkV, dO, seq, nk, sc);
    }
    
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    
    cudaFree(dQ); cudaFree(duK); cudaFree(duV); cudaFree(dkK); cudaFree(dkV); cudaFree(dO);
    return ms;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  AOT LOOP-FREE vs LOOP @ 65536\n");
    printf("  Testing if warp-parallel removes the 128-token floor\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    int seq = 65536;
    
    printf("Know Tokens |  LOOP (ms)  | LOOPFREE (ms) | Speedup\n");
    printf("------------|-------------|---------------|--------\n");
    
    int know_counts[] = {16, 32, 64, 128, 256, 512};
    for (int i = 0; i < 6; i++) {
        int nk = know_counts[i];
        float loop_ms = bench(seq, nk, false);
        float free_ms = bench(seq, nk, true);
        float speedup = loop_ms / free_ms;
        printf("%11d | %9.1f   | %11.1f   | %.2fx\n", nk, loop_ms, free_ms, speedup);
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  Can we go below 64 knowledge tokens now?\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    int tiny[] = {8, 16, 32, 64};
    for (int i = 0; i < 4; i++) {
        int nk = tiny[i];
        float ms = bench(seq, nk, true);
        printf("LOOPFREE @ %d knowledge tokens: %.1f ms\n", nk, ms);
    }
    
    return 0;
}

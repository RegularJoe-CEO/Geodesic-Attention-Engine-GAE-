// True fused kernel - data stays in shared memory between ops
// vs unfused - two separate cuBLAS calls with HBM round-trip

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <nvml.h>

#define D_MODEL 768
#define D_MLP 3072
#define TILE 32

// FUSED: Both ops in one kernel, intermediate stays in SMEM
__global__ void fused_two_layer(
    const half* __restrict__ input,    // [batch*seq, d_model]
    const half* __restrict__ W1,       // [d_model, d_mlp]
    const half* __restrict__ W2,       // [d_mlp, d_model]
    half* __restrict__ output,         // [batch*seq, d_model]
    int N                              // batch*seq
) {
    __shared__ half smem_h1[TILE][D_MLP];  // Intermediate stays HERE, not HBM
    
    int row = blockIdx.x * TILE + threadIdx.y;
    if (row >= N) return;
    
    // Layer 1: input @ W1 -> h1 (stored in SMEM)
    for (int j = threadIdx.x; j < D_MLP; j += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D_MODEL; k++) {
            acc += __half2float(input[row * D_MODEL + k]) * 
                   __half2float(W1[k * D_MLP + j]);
        }
        // GELU activation
        float x = acc;
        acc = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        smem_h1[threadIdx.y][j] = __float2half(acc);
    }
    __syncthreads();
    
    // Layer 2: h1 @ W2 -> output (h1 read from SMEM, never touched HBM)
    for (int j = threadIdx.x; j < D_MODEL; j += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D_MLP; k++) {
            acc += __half2float(smem_h1[threadIdx.y][k]) * 
                   __half2float(W2[k * D_MODEL + j]);
        }
        output[row * D_MODEL + j] = __float2half(acc);
    }
}

int main() {
    // This is the skeleton - full benchmark would:
    // 1. Run fused_two_layer kernel
    // 2. Run two separate cuBLAS calls
    // 3. Compare power/throughput
    
    printf("CUDA fused kernel template ready\n");
    printf("Intermediate h1 [%d x %d] stays in shared memory\n", TILE, D_MLP);
    printf("HBM round-trip eliminated: %d bytes per tile\n", TILE * D_MLP * 2);
    return 0;
}

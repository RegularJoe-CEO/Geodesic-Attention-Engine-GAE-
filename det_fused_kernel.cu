#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define D_MODEL 768
#define D_MLP 3072
#define TILE_M 4  // 4 * 3072 * 4 = 49152 bytes = exactly 48KB SMEM limit

__device__ __forceinline__ float gelu(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// FUSED: Two layers, h1 stays in shared memory (DETERMINISTIC)
__global__ void fused_two_layer_deterministic(
    const float* __restrict__ input,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    float* __restrict__ output,
    int N
) {
    __shared__ float smem_h1[TILE_M][D_MLP];  // NULL-SPACE BUFFER - never touches HBM
    
    int tile_start = blockIdx.x * TILE_M;
    int local_row = threadIdx.y;
    int global_row = tile_start + local_row;
    
    if (global_row >= N) return;
    
    // LAYER 1: input @ W1 -> GELU -> smem_h1
    for (int j = threadIdx.x; j < D_MLP; j += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D_MODEL; k++) {
            acc += input[global_row * D_MODEL + k] * W1[k * D_MLP + j];
        }
        smem_h1[local_row][j] = gelu(acc);
    }
    __syncthreads();
    
    // LAYER 2: smem_h1 @ W2 -> output (h1 read from SMEM, not HBM)
    for (int j = threadIdx.x; j < D_MODEL; j += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D_MLP; k++) {
            acc += smem_h1[local_row][k] * W2[k * D_MODEL + j];
        }
        output[global_row * D_MODEL + j] = acc;
    }
}

// UNFUSED: Two kernels with HBM round-trip
__global__ void layer1_only(const float* input, const float* W1, float* h1, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D_MLP; j += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D_MODEL; k++) acc += input[row * D_MODEL + k] * W1[k * D_MLP + j];
        h1[row * D_MLP + j] = gelu(acc);
    }
}

__global__ void layer2_only(const float* h1, const float* W2, float* output, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D_MODEL; j += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D_MLP; k++) acc += h1[row * D_MLP + k] * W2[k * D_MODEL + j];
        output[row * D_MODEL + j] = acc;
    }
}

int main() {
    int N = 16 * 256;  // batch * seq = 4096
    
    printf("Deterministic Linear Algebra Engine - Fused Kernel Test\n");
    printf("=========================================================\n");
    printf("N=%d, D_MODEL=%d, D_MLP=%d\n", N, D_MODEL, D_MLP);
    printf("SMEM: %d bytes (limit 49152)\n\n", TILE_M * D_MLP * (int)sizeof(float));
    
    float *d_in, *d_W1, *d_W2, *d_out, *d_h1;
    cudaMalloc(&d_in, N * D_MODEL * sizeof(float));
    cudaMalloc(&d_W1, D_MODEL * D_MLP * sizeof(float));
    cudaMalloc(&d_W2, D_MLP * D_MODEL * sizeof(float));
    cudaMalloc(&d_out, N * D_MODEL * sizeof(float));
    cudaMalloc(&d_h1, N * D_MLP * sizeof(float));
    
    // Init
    float* h_data = (float*)malloc(D_MODEL * D_MLP * sizeof(float));
    srand(42);
    for (int i = 0; i < N * D_MODEL; i++) { float v = rand()/(float)RAND_MAX*0.1f; cudaMemcpy(d_in+i,&v,4,cudaMemcpyHostToDevice); }
    for (int i = 0; i < D_MODEL * D_MLP; i++) { h_data[i] = rand()/(float)RAND_MAX*0.1f; }
    cudaMemcpy(d_W1, h_data, D_MODEL*D_MLP*4, cudaMemcpyHostToDevice);
    for (int i = 0; i < D_MLP * D_MODEL; i++) { h_data[i] = rand()/(float)RAND_MAX*0.1f; }
    cudaMemcpy(d_W2, h_data, D_MLP*D_MODEL*4, cudaMemcpyHostToDevice);
    
    // Test determinism
    printf("Testing DETERMINISM (10 runs)...\n");
    float* ref = (float*)malloc(N * D_MODEL * sizeof(float));
    float* test = (float*)malloc(N * D_MODEL * sizeof(float));
    dim3 blk(32, TILE_M), grd((N+TILE_M-1)/TILE_M);
    
    fused_two_layer_deterministic<<<grd,blk>>>(d_in, d_W1, d_W2, d_out, N);
    cudaMemcpy(ref, d_out, N*D_MODEL*4, cudaMemcpyDeviceToHost);
    
    int deterministic = 1;
    for (int r = 0; r < 10; r++) {
        fused_two_layer_deterministic<<<grd,blk>>>(d_in, d_W1, d_W2, d_out, N);
        cudaMemcpy(test, d_out, N*D_MODEL*4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N*D_MODEL; i++) {
            if (test[i] != ref[i]) { deterministic = 0; break; }
        }
    }
    printf("%s BIT-EXACT across 10 runs\n\n", deterministic ? "✓" : "✗");
    
    // Warmup
    for (int i = 0; i < 200; i++) {
        fused_two_layer_deterministic<<<grd,blk>>>(d_in, d_W1, d_W2, d_out, N);
        layer1_only<<<(N+7)/8, dim3(32,8)>>>(d_in, d_W1, d_h1, N);
        layer2_only<<<(N+7)/8, dim3(32,8)>>>(d_h1, d_W2, d_out, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int iters = 1000;
    
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) fused_two_layer_deterministic<<<grd,blk>>>(d_in, d_W1, d_W2, d_out, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float fused_ms; cudaEventElapsedTime(&fused_ms, t0, t1);
    
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) {
        layer1_only<<<(N+7)/8, dim3(32,8)>>>(d_in, d_W1, d_h1, N);
        layer2_only<<<(N+7)/8, dim3(32,8)>>>(d_h1, d_W2, d_out, N);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float unfused_ms; cudaEventElapsedTime(&unfused_ms, t0, t1);
    
    printf("========== PERFORMANCE ==========\n");
    printf("FUSED (deterministic):   %.2f ms (%d iters)\n", fused_ms, iters);
    printf("UNFUSED (HBM roundtrip): %.2f ms (%d iters)\n", unfused_ms, iters);
    printf("Speedup: %.2fx\n\n", unfused_ms / fused_ms);
    
    printf("========== MEMORY TRAFFIC ==========\n");
    float fused_mb = N * D_MODEL * 4 * 2 / 1e6;  // in + out only
    float unfused_mb = fused_mb + N * D_MLP * 4 * 2 / 1e6;  // + h1 write/read
    printf("FUSED:   %.1f MB (h1 in SMEM)\n", fused_mb);
    printf("UNFUSED: %.1f MB (h1 via HBM)\n", unfused_mb);
    printf("SAVED:   %.1f MB (%.0f%% reduction)\n", unfused_mb-fused_mb, 100*(unfused_mb-fused_mb)/unfused_mb);
    
    return 0;
}

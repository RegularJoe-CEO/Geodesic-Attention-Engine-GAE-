#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define D_MODEL 768
#define D_MLP 3072
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ __forceinline__ half gelu_half(half x) {
    float xf = __half2float(x);
    float g = xf * 0.5f * (1.0f + tanhf(0.7978845608f * (xf + 0.044715f * xf * xf * xf)));
    return __float2half(g);
}

// Optimized fused kernel with Tensor Cores
// Uses WMMA for matrix multiply, keeps intermediate in shared memory
__global__ void fused_tensor_core(
    const half* __restrict__ input,    // [N, D_MODEL]
    const half* __restrict__ W1,       // [D_MODEL, D_MLP]  
    const half* __restrict__ W2,       // [D_MLP, D_MODEL]
    half* __restrict__ output,         // [N, D_MODEL]
    int N
) {
    // Each block handles WMMA_M rows
    // Shared memory for intermediate h1 - stays here between layer 1 and 2
    __shared__ half smem_h1[WMMA_M][D_MLP + 8];  // +8 for bank conflict avoidance
    __shared__ half smem_input[WMMA_M][D_MODEL + 8];
    __shared__ half smem_W1_tile[D_MODEL][WMMA_N + 8];
    
    int block_row = blockIdx.x * WMMA_M;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (block_row >= N) return;
    
    // Load input tile to shared memory (coalesced)
    for (int i = tid; i < WMMA_M * D_MODEL; i += blockDim.x) {
        int r = i / D_MODEL;
        int c = i % D_MODEL;
        if (block_row + r < N)
            smem_input[r][c] = input[(block_row + r) * D_MODEL + c];
    }
    __syncthreads();
    
    // LAYER 1: input @ W1 -> GELU -> smem_h1
    // Process D_MLP columns in tiles of WMMA_N
    for (int col_tile = 0; col_tile < D_MLP; col_tile += WMMA_N) {
        // Declare fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
        
        wmma::fill_fragment(c_frag, __float2half(0.0f));
        
        // Accumulate over K dimension
        for (int k = 0; k < D_MODEL; k += WMMA_K) {
            // Load A tile (input)
            wmma::load_matrix_sync(a_frag, &smem_input[0][k], D_MODEL + 8);
            // Load B tile (W1) - directly from global for now
            wmma::load_matrix_sync(b_frag, &W1[k * D_MLP + col_tile], D_MLP);
            // Multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        // Store to shared memory with GELU
        wmma::store_matrix_sync(&smem_h1[0][col_tile], c_frag, D_MLP + 8, wmma::mem_row_major);
    }
    __syncthreads();
    
    // Apply GELU in shared memory
    for (int i = tid; i < WMMA_M * D_MLP; i += blockDim.x) {
        int r = i / D_MLP;
        int c = i % D_MLP;
        smem_h1[r][c] = gelu_half(smem_h1[r][c]);
    }
    __syncthreads();
    
    // LAYER 2: smem_h1 @ W2 -> output
    // h1 is already in shared memory - NO HBM ROUND TRIP
    for (int col_tile = 0; col_tile < D_MODEL; col_tile += WMMA_N) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
        
        wmma::fill_fragment(c_frag, __float2half(0.0f));
        
        for (int k = 0; k < D_MLP; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, &smem_h1[0][k], D_MLP + 8);
            wmma::load_matrix_sync(b_frag, &W2[k * D_MODEL + col_tile], D_MODEL);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        // Store directly to global output
        wmma::store_matrix_sync(&output[block_row * D_MODEL + col_tile], c_frag, D_MODEL, wmma::mem_row_major);
    }
}

// Reference unfused using cuBLAS-style separate kernels
__global__ void layer1_naive(const half* in, const half* W1, half* h1, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D_MLP; j += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D_MODEL; k++)
            acc += __half2float(in[row*D_MODEL+k]) * __half2float(W1[k*D_MLP+j]);
        float g = acc * 0.5f * (1.0f + tanhf(0.7978845608f * (acc + 0.044715f*acc*acc*acc)));
        h1[row*D_MLP+j] = __float2half(g);
    }
}

__global__ void layer2_naive(const half* h1, const half* W2, half* out, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D_MODEL; j += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D_MLP; k++)
            acc += __half2float(h1[row*D_MLP+k]) * __half2float(W2[k*D_MODEL+j]);
        out[row*D_MODEL+j] = __float2half(acc);
    }
}

int main() {
    int N = 4096;
    
    printf("Tensor Core Fused Kernel - Deterministic Linear Algebra Engine\n");
    printf("===============================================================\n");
    printf("N=%d, D_MODEL=%d, D_MLP=%d\n", N, D_MODEL, D_MLP);
    printf("Using WMMA %dx%dx%d\n\n", WMMA_M, WMMA_N, WMMA_K);
    
    half *d_in, *d_W1, *d_W2, *d_out, *d_h1;
    cudaMalloc(&d_in, N * D_MODEL * sizeof(half));
    cudaMalloc(&d_W1, D_MODEL * D_MLP * sizeof(half));
    cudaMalloc(&d_W2, D_MLP * D_MODEL * sizeof(half));
    cudaMalloc(&d_out, N * D_MODEL * sizeof(half));
    cudaMalloc(&d_h1, N * D_MLP * sizeof(half));
    
    // Init random
    half* h_tmp = (half*)malloc(D_MODEL * D_MLP * sizeof(half));
    srand(42);
    for (int i = 0; i < N * D_MODEL; i++) {
        half v = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.1f);
        cudaMemcpy(d_in + i, &v, sizeof(half), cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < D_MODEL * D_MLP; i++) h_tmp[i] = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.1f);
    cudaMemcpy(d_W1, h_tmp, D_MODEL * D_MLP * sizeof(half), cudaMemcpyHostToDevice);
    for (int i = 0; i < D_MLP * D_MODEL; i++) h_tmp[i] = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.1f);
    cudaMemcpy(d_W2, h_tmp, D_MLP * D_MODEL * sizeof(half), cudaMemcpyHostToDevice);
    
    // Test determinism
    printf("Testing DETERMINISM...\n");
    half* ref = (half*)malloc(N * D_MODEL * sizeof(half));
    half* test = (half*)malloc(N * D_MODEL * sizeof(half));
    
    dim3 blk_tc(128);  // 4 warps
    dim3 grd_tc((N + WMMA_M - 1) / WMMA_M);
    
    fused_tensor_core<<<grd_tc, blk_tc>>>(d_in, d_W1, d_W2, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ref, d_out, N * D_MODEL * sizeof(half), cudaMemcpyDeviceToHost);
    
    int deterministic = 1;
    for (int r = 0; r < 10; r++) {
        fused_tensor_core<<<grd_tc, blk_tc>>>(d_in, d_W1, d_W2, d_out, N);
        cudaDeviceSynchronize();
        cudaMemcpy(test, d_out, N * D_MODEL * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N * D_MODEL; i++) {
            if (__half2float(test[i]) != __half2float(ref[i])) {
                printf("  MISMATCH at run %d idx %d: %.6f vs %.6f\n", r, i,
                       __half2float(ref[i]), __half2float(test[i]));
                deterministic = 0;
                break;
            }
        }
        if (!deterministic) break;
    }
    printf("%s BIT-EXACT across 10 runs\n\n", deterministic ? "✓" : "✗");
    
    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 100; i++) {
        fused_tensor_core<<<grd_tc, blk_tc>>>(d_in, d_W1, d_W2, d_out, N);
        layer1_naive<<<(N+7)/8, dim3(32,8)>>>(d_in, d_W1, d_h1, N);
        layer2_naive<<<(N+7)/8, dim3(32,8)>>>(d_h1, d_W2, d_out, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    int iters = 1000;
    
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++)
        fused_tensor_core<<<grd_tc, blk_tc>>>(d_in, d_W1, d_W2, d_out, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float fused_ms;
    cudaEventElapsedTime(&fused_ms, t0, t1);
    
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) {
        layer1_naive<<<(N+7)/8, dim3(32,8)>>>(d_in, d_W1, d_h1, N);
        layer2_naive<<<(N+7)/8, dim3(32,8)>>>(d_h1, d_W2, d_out, N);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float unfused_ms;
    cudaEventElapsedTime(&unfused_ms, t0, t1);
    
    printf("\n========== RESULTS ==========\n");
    printf("FUSED (Tensor Core):     %.2f ms (%d iters) = %.3f ms/iter\n", fused_ms, iters, fused_ms/iters);
    printf("UNFUSED (naive):         %.2f ms (%d iters) = %.3f ms/iter\n", unfused_ms, iters, unfused_ms/iters);
    printf("Speedup: %.2fx\n\n", unfused_ms / fused_ms);
    
    float fused_mb = N * D_MODEL * 2 * 2 / 1e6;
    float unfused_mb = fused_mb + N * D_MLP * 2 * 2 / 1e6;
    printf("Memory: FUSED %.1f MB, UNFUSED %.1f MB, SAVED %.1f MB (%.0f%%)\n",
           fused_mb, unfused_mb, unfused_mb - fused_mb, 100*(unfused_mb-fused_mb)/unfused_mb);
    
    cudaFree(d_in); cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_out); cudaFree(d_h1);
    free(h_tmp); free(ref); free(test);
    
    return 0;
}

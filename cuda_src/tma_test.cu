// Copyright © 2025-2026 Eric Waller. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the repository root.
//
// Original contributions: Deterministic single-kernel fused attention,
// bare-metal Hopper (WGMMA/TMA) execution, energy-efficient design.
//
// Builds on: Scaled dot-product attention (Vaswani et al., 2017),
// Online softmax (Milakov & Gimelshein, 2018),
// Tiled fused attention (Dao et al., 2022).
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define M_TILE 64
#define N_TILE 64
#define D_HEAD 128
#define K_CHUNK 16
#define TILE_B_BYTES (K_CHUNK * N_TILE * 2)
#define SMEM_TOTAL 32768

__device__ __forceinline__
void mbar_init(uint64_t* mbar, int thread_count) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;\n"
        : : "r"(mbar_addr), "r"(thread_count) : "memory");
}

__device__ __forceinline__
void mbar_expect_tx(uint64_t* mbar, int expected_tx_bytes) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        : : "r"(mbar_addr), "r"(expected_tx_bytes) : "memory");
}

__device__ __forceinline__
void mbar_wait(uint64_t* mbar, int phase) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_LOOP;\n"
        "}\n"
        : : "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__
void tma_load_2d(void* smem_ptr, const CUtensorMap* tma_desc, int coord_x, int coord_y, uint64_t* mbar) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(tma_desc);
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :
        : "r"(smem_addr), "l"(tma_ptr), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
        : "memory");
}

__global__ void __launch_bounds__(128, 1)
tma_test_kernel(const half* __restrict__ K,
                float* __restrict__ out,
                const __grid_constant__ CUtensorMap tma_K) {
    extern __shared__ __align__(128) char smem_raw[];
    half* sB = reinterpret_cast<half*>(smem_raw);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + TILE_B_BYTES);

    int tid = threadIdx.x;
    if (tid == 0) {
        mbar_init(mbar, 1);
    }
    __syncthreads();

    if (tid == 0) {
        mbar_expect_tx(mbar, TILE_B_BYTES);
        tma_load_2d(sB, &tma_K, 0, 0, mbar);
    }
    __syncthreads();

    mbar_wait(mbar, 0);
    __syncthreads();

    if (tid == 0) {
        out[0] = __half2float(sB[0]);
    }
}

CUtensorMap create_tma_desc_K(const half* K_ptr, int total_rows, int total_cols) {
    CUtensorMap tmap;
    uint64_t globalDim[2]    = {(uint64_t)total_cols, (uint64_t)total_rows};
    uint64_t globalStride[1] = {(uint64_t)(total_cols * sizeof(half))};
    uint32_t boxDim[2]       = {K_CHUNK, N_TILE};
    uint32_t elemStride[2]   = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        (void*)K_ptr,
        globalDim,
        globalStride,
        boxDim,
        elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        printf("cuTensorMapEncodeTiled FAILED: %s\n", errStr);
    } else {
        printf("TMA descriptor created OK\n");
    }
    return tmap;
}

int main() {
    cuInit(0);

    int total_rows = N_TILE;
    int total_cols = D_HEAD;
    size_t elems = (size_t)total_rows * total_cols;
    size_t bytes = elems * sizeof(half);

    half* hK = (half*)malloc(bytes);
    for (size_t i = 0; i < elems; i++) {
        hK[i] = __float2half(1.0f);
    }

    half* dK = nullptr;
    float* dOut = nullptr;
    cudaMalloc(&dK, bytes);
    cudaMalloc(&dOut, sizeof(float));
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);

    CUtensorMap h_tma_K = create_tma_desc_K(dK, total_rows, total_cols);
    cudaDeviceSynchronize();

    tma_test_kernel<<<1, 128, SMEM_TOTAL>>>(dK, dOut, h_tma_K);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float hOut = 0.0f;
    cudaMemcpy(&hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost);
    printf("TMA test output: %.6f\n", hOut);

    cudaFree(dK);
    cudaFree(dOut);
    free(hK);
    return 0;
}

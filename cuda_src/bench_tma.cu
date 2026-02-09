// TMA bring-up: single warpgroup, TMA for sB (K operand) only
// Everything else identical to the 30 TFLOPS baseline
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Device constant tensor map (safe address space for TMA)


#define M_TILE    64
#define N_TILE    64
#define D_HEAD    128
#define K_CHUNK   16
#define N_CHUNKS  (D_HEAD / K_CHUNK)
#define WARPGROUP 128
#define BLOCK_THREADS 256
#define PV_K_CHUNKS (N_TILE / K_CHUNK)
#define O_COL_TILES (D_HEAD / N_TILE)
#define TILE_A_BYTES (K_CHUNK * M_TILE * 2)
#define TILE_B_BYTES (K_CHUNK * N_TILE * 2)
#define SMEM_Q_BYTES (M_TILE * D_HEAD * 2)
#define SMEM_TOTAL   32768
#define DESC_LEAD    128
#define DESC_STRIDE  1024
#define SWIZZLE_B128 3

__device__ __forceinline__
uint32_t swizzle_b128(uint32_t row, uint32_t col) {
    uint32_t off = row * 128 + col * 2;
    return off ^ ((off & 0x380) >> 3);
}

__device__ __forceinline__
uint64_t make_desc(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t d = 0;
    d |= ((uint64_t)(addr & 0x3FFFF) >> 4) << 0;
    d |= ((uint64_t)(DESC_LEAD)  >> 4) << 16;
    d |= ((uint64_t)(DESC_STRIDE) >> 4) << 32;
    d |= ((uint64_t)SWIZZLE_B128) << 62;
    return d;
}

__device__ __forceinline__
void wgmma_m64n64k16(float acc[32], uint64_t dA, uint64_t dB) {
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,1,1,1,0,1;\n"
        "}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
          "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
          "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
          "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
          "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
        : "l"(dA), "l"(dB));
}

__device__ __forceinline__
void store_p_chunk_to_smem(half* sA, float* p_local, int kc, int tid) {
    int W = tid / 32;
    int L = tid % 32;
    int base_m = W * 16;
    int r_in_group = L / 4;
    int col_pair   = L % 4;
    for (int rr = 0; rr < 2; rr++) {
        int m = base_m + r_in_group + rr * 8;
        int j0 = kc * K_CHUNK + col_pair * 2;
        int reg0 = rr * 16 + col_pair * 2;
        int reg1 = reg0 + 1;
        uint32_t dst = swizzle_b128(j0 - kc * K_CHUNK, m);
        sA[dst / 2]     = __float2half(p_local[reg0]);
        sA[dst / 2 + 1] = __float2half(p_local[reg1]);
    }
}

// TMA load: single thread issues cp.async.bulk.tensor.2d
// smem_ptr must be 128-byte aligned for TMA
// tma_desc is the CUtensorMap passed via constant memory or kernel arg
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

// mbarrier init, arrive, wait
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
    // Spin wait on phase
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
void mbar_arrive(uint64_t* mbar) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.arrive.shared.b64 _, [%0];\n"
        : : "r"(mbar_addr) : "memory");
}

// Kernel: same as 30 TFLOPS baseline but sB loaded via TMA
__global__ void __launch_bounds__(256, 1)
gae_tma_kernel(const half* __restrict__ Q,
               const half* __restrict__ K,
               const half* __restrict__ V,
               float* __restrict__ O,
               const __grid_constant__ CUtensorMap tma_K,
               const __grid_constant__ CUtensorMap tma_V,
                 int seq_len, int num_heads) {
    int row_tile = blockIdx.x;
    int head     = blockIdx.y;
    int batch    = blockIdx.z;

    long long bh = (long long)(batch * num_heads + head);
    long long qkv_off = bh * seq_len * D_HEAD;

    const half* myQ = Q + qkv_off + (long long)row_tile * M_TILE * D_HEAD;
    const half* myV = V + qkv_off;
    float* myO      = O + qkv_off + (long long)row_tile * M_TILE * D_HEAD;

    extern __shared__ char smem_raw[];
    // Layout: sQ(16384) | sA(2048) | sB0(2048) | sB1(2048) | mbar0(16) | mbar1(16) | mbar2(16) | mbar3(16)
    // sB buffers must be 128B aligned
    // sQ_full at offset 0: 16384 bytes (128-aligned: yes)
    // sA at offset 16384: 2048 bytes
    // sB0 at offset 18432: 2048 bytes (18432 = 144*128, 128-aligned: yes)
    // sB1 at offset 20480: 2048 bytes (20480 = 160*128, 128-aligned: yes)
    // mbar0 at offset 22528: 16 bytes
    // mbar1 at offset 22544: 16 bytes
    // mbar2 at offset 22560: 16 bytes
    // mbar3 at offset 22576: 16 bytes
    half* sQ_full = reinterpret_cast<half*>(smem_raw);
    half* sA      = reinterpret_cast<half*>(smem_raw + SMEM_Q_BYTES);
    half* sB0     = reinterpret_cast<half*>(smem_raw + SMEM_Q_BYTES + TILE_A_BYTES);
    half* sB1     = reinterpret_cast<half*>(smem_raw + SMEM_Q_BYTES + TILE_A_BYTES + TILE_B_BYTES);
    struct alignas(16) Mbarrier {
        uint64_t data[2];
    };
    Mbarrier* mbarriers = reinterpret_cast<Mbarrier*>(
        smem_raw + SMEM_Q_BYTES + TILE_A_BYTES + 2 * TILE_B_BYTES);
    uint64_t* mbar0 = mbarriers[0].data;
    uint64_t* mbar1 = mbarriers[1].data;
    uint64_t* mbar2 = mbarriers[2].data;
    uint64_t* mbar3 = mbarriers[3].data;

    int tid = threadIdx.x;
    bool is_compute = tid < WARPGROUP;
    bool is_tma = tid == WARPGROUP;
    int W   = tid / 32;
    int L   = tid % 32;
    int col_tiles = seq_len / N_TILE;

    // Init mbarriers (thread 0 only) - count=1, arrive once to set initial phase
    if (tid == 0) {
        mbar_init(mbar0, 1);
        mbar_init(mbar1, 1);
        mbar_init(mbar2, 1);
        mbar_init(mbar3, 1);
    }
    __syncthreads();

    // Load Q into sQ_full (swizzled, one-time)
    if (is_compute) {
        for (int c = 0; c < N_CHUNKS; c++) {
            for (int i = tid; i < K_CHUNK * M_TILE; i += WARPGROUP) {
                int row = i / K_CHUNK;
                int col = i % K_CHUNK;
                uint32_t dst = swizzle_b128(col, row);
                sQ_full[c * (K_CHUNK * M_TILE) + dst / 2] = myQ[row * D_HEAD + c * K_CHUNK + col];
            }
        }
    }
    __syncthreads();

    float row_max_0 = -1e30f;
    float row_max_1 = -1e30f;
    float row_sum_0 = 0.0f;
    float row_sum_1 = 0.0f;

    float o_acc[O_COL_TILES][32];
    if (is_compute) {
        for (int ot = 0; ot < O_COL_TILES; ot++)
            for (int i = 0; i < 32; i++)
                o_acc[ot][i] = 0.0f;
    }

    for (int ct = 0; ct < col_tiles; ct++) {
        const half* myV_tile = myV + (long long)ct * N_TILE * D_HEAD;

        // Phase 1: S = Q * K^T
        float s_accum[32];
        if (is_compute) {
            for (int i = 0; i < 32; i++) s_accum[i] = 0.0f;
        }

        int phase0 = 0;
        int phase1 = 0;
        int cur = 0;
        int next = 1;

        if (is_tma) {
            mbar_expect_tx(mbar0, TILE_B_BYTES);
            tma_load_2d(sB0, &tma_K, 0, (int)(bh * seq_len + ct * N_TILE), mbar0);
        }
        __syncthreads();

        for (int c = 0; c < N_CHUNKS; c++) {
            half* sB_cur = (cur == 0) ? sB0 : sB1;
            uint64_t* mbar_cur = (cur == 0) ? mbar0 : mbar1;
            uint64_t* mbar_next = (next == 0) ? mbar0 : mbar1;

            if (is_compute) {
                mbar_wait(mbar_cur, (cur == 0) ? phase0 : phase1);
                if (cur == 0) {
                    phase0 ^= 1;
                } else {
                    phase1 ^= 1;
                }
            }

            if (is_tma && c + 1 < N_CHUNKS) {
                mbar_expect_tx(mbar_next, TILE_B_BYTES);
                tma_load_2d((next == 0) ? sB0 : sB1,
                            &tma_K,
                            (c + 1) * K_CHUNK,
                            (int)(bh * seq_len + ct * N_TILE),
                            mbar_next);
            }

            if (is_compute) {
                half* sQ_chunk = sQ_full + c * (K_CHUNK * M_TILE);

                asm volatile("wgmma.fence.sync.aligned;\n");
                uint64_t dA = make_desc(sQ_chunk);
                uint64_t dB = make_desc(sB_cur);
                wgmma_m64n64k16(s_accum, dA, dB);
                asm volatile("wgmma.commit_group.sync.aligned;\n");
                asm volatile("wgmma.wait_group.sync.aligned 0;\n");
            }
            cur ^= 1;
            next ^= 1;
        }

        float p_local[32];
        if (is_compute) {
            #pragma unroll
            for (int i = 0; i < 32; i++) p_local[i] = s_accum[i];

            // Phase 2: Online softmax
            float local_max_0 = -1e30f;
            float local_max_1 = -1e30f;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                local_max_0 = fmaxf(local_max_0, p_local[i]);
                local_max_1 = fmaxf(local_max_1, p_local[16 + i]);
            }
            for (int mask = 2; mask >= 1; mask >>= 1) {
                local_max_0 = fmaxf(local_max_0, __shfl_xor_sync(0xffffffff, local_max_0, mask));
                local_max_1 = fmaxf(local_max_1, __shfl_xor_sync(0xffffffff, local_max_1, mask));
            }

            float new_max_0 = fmaxf(row_max_0, local_max_0);
            float new_max_1 = fmaxf(row_max_1, local_max_1);
            float scale_0 = expf(row_max_0 - new_max_0);
            float scale_1 = expf(row_max_1 - new_max_1);

            row_sum_0 *= scale_0;
            row_sum_1 *= scale_1;
            #pragma unroll
            for (int ot = 0; ot < O_COL_TILES; ot++) {
                for (int i = 0; i < 16; i++) {
                    o_acc[ot][i]      *= scale_0;
                    o_acc[ot][16 + i] *= scale_1;
                }
            }

            float local_sum_0 = 0.0f;
            float local_sum_1 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                p_local[i]      = expf(p_local[i]      - new_max_0);
                p_local[16 + i] = expf(p_local[16 + i] - new_max_1);
                local_sum_0 += p_local[i];
                local_sum_1 += p_local[16 + i];
            }
            for (int mask = 2; mask >= 1; mask >>= 1) {
                local_sum_0 += __shfl_xor_sync(0xffffffff, local_sum_0, mask);
                local_sum_1 += __shfl_xor_sync(0xffffffff, local_sum_1, mask);
            }

            row_sum_0 += local_sum_0;
            row_sum_1 += local_sum_1;
            row_max_0 = new_max_0;
            row_max_1 = new_max_1;
        }

        // Phase 3: O += P * V (TMA double-buffer for V)
        for (int ot = 0; ot < O_COL_TILES; ot++) {
            int v_phase0 = 0;
            int v_phase1 = 0;
            int v_cur = 0;
            int v_next = 1;

            if (is_tma) {
                mbar_expect_tx(mbar2, TILE_B_BYTES);
                tma_load_2d(sB0,
                            &tma_V,
                            ot * N_TILE,
                            (int)(bh * seq_len + ct * N_TILE),
                            mbar2);
            }
            __syncthreads();

            for (int kc = 0; kc < PV_K_CHUNKS; kc++) {
                if (is_compute) {
                    store_p_chunk_to_smem(sA, p_local, kc, tid);
                    asm volatile("bar.sync 1, 128;\n");
                }

                half* sB_cur = (v_cur == 0) ? sB0 : sB1;
                uint64_t* v_mbar_cur = (v_cur == 0) ? mbar2 : mbar3;
                uint64_t* v_mbar_next = (v_next == 0) ? mbar2 : mbar3;

                if (is_compute) {
                    mbar_wait(v_mbar_cur, (v_cur == 0) ? v_phase0 : v_phase1);
                    if (v_cur == 0) {
                        v_phase0 ^= 1;
                    } else {
                        v_phase1 ^= 1;
                    }
                }

                if (is_tma && kc + 1 < PV_K_CHUNKS) {
                    mbar_expect_tx(v_mbar_next, TILE_B_BYTES);
                    tma_load_2d((v_next == 0) ? sB0 : sB1,
                                &tma_V,
                                ot * N_TILE,
                                (int)(bh * seq_len + ct * N_TILE + (kc + 1) * K_CHUNK),
                                v_mbar_next);
                }

                if (is_compute) {
                    asm volatile("wgmma.fence.sync.aligned;\n");
                    uint64_t dA2 = make_desc(sA);
                    uint64_t dB2 = make_desc(sB_cur);
                    wgmma_m64n64k16(o_acc[ot], dA2, dB2);
                    asm volatile("wgmma.commit_group.sync.aligned;\n");
                    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
                }

                v_cur ^= 1;
                v_next ^= 1;
            }
        }
    }

    // Final normalization
    if (is_compute) {
        float inv_sum_0 = 1.0f / row_sum_0;
        float inv_sum_1 = 1.0f / row_sum_1;

        for (int ot = 0; ot < O_COL_TILES; ot++) {
            int base_m = W * 16;
            int r_in_group = L / 4;
            int col_pair   = L % 4;
            for (int rr = 0; rr < 2; rr++) {
                int m = base_m + r_in_group + rr * 8;
                float inv = (rr == 0) ? inv_sum_0 : inv_sum_1;
                for (int cp = 0; cp < 2; cp++) {
                    int col = col_pair * 2 + cp;
                    int reg_idx = rr * 16 + col_pair * 2 + cp;
                    if (reg_idx < 32 && m < M_TILE)
                        myO[m * D_HEAD + ot * N_TILE + col] = o_acc[ot][reg_idx] * inv;
                }
            }
        }
    }
}

// Host: create TMA descriptor for K tensor
CUtensorMap create_tma_desc_K(const half* K_ptr, int total_rows, int total_cols) {
    CUtensorMap tmap;
    // K is [total_rows x total_cols] row-major, fp16
    // total_rows = batch * heads * seq_len
    // total_cols = D_HEAD = 128
    // TMA tile: K_CHUNK cols x N_TILE rows = 16 x 64
    // Global dims: [total_cols, total_rows] (TMA uses column-major convention)
    // Global strides: [sizeof(half), total_cols * sizeof(half)] = [2, 256]
    // Box dims: [K_CHUNK, N_TILE] = [16, 64]
    // Swizzle: 128B

    uint64_t globalDim[2]    = {(uint64_t)total_cols, (uint64_t)total_rows};
    uint64_t globalStride[1] = {(uint64_t)(total_cols * sizeof(half))}; // stride for dim1 (row stride in bytes), dim0 stride is implicit (element size)
    // Actually cuTensorMapEncodeTiled wants globalStrides starting from dim1
    // globalStrides[0] = stride of dim 1 in bytes = total_cols * 2
    uint32_t boxDim[2]       = {K_CHUNK, N_TILE};  // 16 x 64
    uint32_t elemStride[2]   = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,                                    // rank
        (void*)K_ptr,                         // global address
        globalDim,                            // global dimensions
        globalStride,                         // global strides (bytes, starting from dim 1)
        boxDim,                               // box dimensions
        elemStride,                           // element strides
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

// Host: create TMA descriptor for V tensor
CUtensorMap create_tma_desc_V(const half* V_ptr, int total_rows, int total_cols) {
    CUtensorMap tmap;
    // V is [total_rows x total_cols] row-major, fp16
    // TMA tile: N_TILE cols x K_CHUNK rows = 64 x 16
    // Global dims: [total_cols, total_rows] (TMA uses column-major convention)
    // Global strides: [sizeof(half), total_cols * sizeof(half)] = [2, 256]
    // Box dims: [N_TILE, K_CHUNK] = [64, 16]
    // Swizzle: 128B

    uint64_t globalDim[2]    = {(uint64_t)total_cols, (uint64_t)total_rows};
    uint64_t globalStride[1] = {(uint64_t)(total_cols * sizeof(half))};
    uint32_t boxDim[2]       = {N_TILE, K_CHUNK};  // 64 x 16
    uint32_t elemStride[2]   = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,                                    // rank
        (void*)V_ptr,                         // global address
        globalDim,                            // global dimensions
        globalStride,                         // global strides (bytes, starting from dim 1)
        boxDim,                               // box dimensions
        elemStride,                           // element strides
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

    int seq_lens[] = {512, 1024, 2048, 4096};
    int num_seq = 4;
    int heads = 32;
    int batch = 1;
    int warmup = 10;
    int runs = 100;

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     GAE TMA Bring-up: sB via TMA, sA thread-level      ║\n");
    printf("║     Gate A: prove cp.async.bulk.tensor in SASS          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("Config: batch=%d  heads=%d  d_head=%d  warmup=%d  runs=%d\n\n",
           batch, heads, D_HEAD, warmup, runs);
    printf("%-10s %8s %10s %10s %10s\n", "seq_len", "blocks", "avg(us)", "TFLOPS", "util%");
    printf("────────────────────────────────────────────────────────────\n");

    for (int si = 0; si < num_seq; si++) {
        int S = seq_lens[si];
        int row_tiles = S / M_TILE;
        int total_blocks = row_tiles * heads * batch;

        size_t qkv_elems = (size_t)batch * heads * S * D_HEAD;
        size_t qkv_bytes = qkv_elems * sizeof(half);
        size_t o_bytes   = qkv_elems * sizeof(float);

        int total_rows = batch * heads * S;
        int total_cols = D_HEAD;

        half *dQ, *dK, *dV;
        float *dO;
        cudaMalloc(&dQ, qkv_bytes);
        cudaMalloc(&dK, qkv_bytes);
        cudaMalloc(&dV, qkv_bytes);
        cudaMalloc(&dO, o_bytes);

        half* hBuf = (half*)malloc(qkv_bytes);
        srand(42 + si);
        for (size_t i = 0; i < qkv_elems; i++)
            hBuf[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
        cudaMemcpy(dQ, hBuf, qkv_bytes, cudaMemcpyHostToDevice);
        for (size_t i = 0; i < qkv_elems; i++)
            hBuf[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
        cudaMemcpy(dK, hBuf, qkv_bytes, cudaMemcpyHostToDevice);
        for (size_t i = 0; i < qkv_elems; i++)
            hBuf[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
        cudaMemcpy(dV, hBuf, qkv_bytes, cudaMemcpyHostToDevice);
        free(hBuf);

        // Create TMA descriptors for K and V
        CUtensorMap h_tma_K = create_tma_desc_K(dK, total_rows, total_cols);
        CUtensorMap h_tma_V = create_tma_desc_V(dV, total_rows, total_cols);
        cudaDeviceSynchronize();


        dim3 grid(row_tiles, heads, batch);
        dim3 block(BLOCK_THREADS);

        for (int i = 0; i < warmup; i++)
            gae_tma_kernel<<<grid, block, SMEM_TOTAL>>>(dQ, dK, dV, dO, h_tma_K, h_tma_V, S, heads);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("%-10d  CUDA ERROR: %s\n", S, cudaGetErrorString(err));
            cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
            continue;
        }

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < runs; i++)
            gae_tma_kernel<<<grid, block, SMEM_TOTAL>>>(dQ, dK, dV, dO, h_tma_K, h_tma_V, S, heads);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float total_ms;
        cudaEventElapsedTime(&total_ms, t0, t1);
        float avg_us = (total_ms / runs) * 1000.0f;

        double flops = 4.0 * batch * heads * (double)S * S * D_HEAD;
        double tflops = (flops / (avg_us * 1e-6)) / 1e12;
        double util = (tflops / 989.0) * 100.0;

        printf("%-10d %8d %10.1f %10.2f %9.1f%%\n",
               S, total_blocks, avg_us, tflops, util);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    printf("────────────────────────────────────────────────────────────\n");
    printf("H100 SXM FP16 peak: 989 TFLOPS\n");
    return 0;
}

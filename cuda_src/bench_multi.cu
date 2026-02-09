#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define M_TILE    64
#define N_TILE    64
#define D_HEAD    128
#define K_CHUNK   16
#define N_CHUNKS  (D_HEAD / K_CHUNK)
#define PV_K_CHUNKS (N_TILE / K_CHUNK)
#define O_COL_TILES (D_HEAD / N_TILE)
#define TILE_BYTES  (K_CHUNK * M_TILE * 2)  // 2048
#define SMEM_Q_BYTES (M_TILE * D_HEAD * 2)  // 16384
// Double-buffered: sA[2] + sB[2] + sQ
// 2*2048 + 2*2048 + 16384 = 24576
#define SMEM_TOTAL   32768
#define DESC_LEAD    128
#define DESC_STRIDE  1024
#define SWIZZLE_B128 3
#define NUM_THREADS  256

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
void store_p_chunk_to_smem(half* sA, float* p_local, int kc, int lane128) {
    int W = lane128 / 32;
    int L = lane128 % 32;
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

// Load K_CHUNK x ROWS tile into swizzled smem from global
__device__ __forceinline__
void load_tile_swizzled(half* dst, const half* src, int rows, int stride,
                        int chunk_idx, int tid, int nthreads) {
    for (int i = tid; i < K_CHUNK * rows; i += nthreads) {
        int row = i / K_CHUNK;
        int col = i % K_CHUNK;
        dst[swizzle_b128(col, row) / 2] = src[row * stride + chunk_idx * K_CHUNK + col];
    }
}

// ============================================================
// 2-warpgroup pipeline: WG0=compute, WG1=producer (async load)
// Double-buffered sA[2], sB[2], Q cached in smem
// ============================================================
__global__ void __launch_bounds__(256, 1)
gae_multi_kernel(const half* __restrict__ Q,
                 const half* __restrict__ K,
                 const half* __restrict__ V,
                 float* __restrict__ O,
                 int seq_len, int num_heads) {
    int row_tile = blockIdx.x;
    int head     = blockIdx.y;
    int batch    = blockIdx.z;

    long long bh = (long long)(batch * num_heads + head);
    long long qkv_off = bh * seq_len * D_HEAD;

    const half* myQ = Q + qkv_off + (long long)row_tile * M_TILE * D_HEAD;
    const half* myK = K + qkv_off;
    const half* myV = V + qkv_off;
    float* myO      = O + qkv_off + (long long)row_tile * M_TILE * D_HEAD;

    extern __shared__ char smem_raw[];
    // Layout: sQ(16384) | sA0(2048) | sA1(2048) | sB0(2048) | sB1(2048)
    half* sQ_full = reinterpret_cast<half*>(smem_raw);
    half* sA0     = reinterpret_cast<half*>(smem_raw + SMEM_Q_BYTES);
    half* sA1     = reinterpret_cast<half*>(smem_raw + SMEM_Q_BYTES + TILE_BYTES);
    half* sB0     = reinterpret_cast<half*>(smem_raw + SMEM_Q_BYTES + 2 * TILE_BYTES);
    half* sB1     = reinterpret_cast<half*>(smem_raw + SMEM_Q_BYTES + 3 * TILE_BYTES);
    half* sA_buf[2] = {sA0, sA1};
    half* sB_buf[2] = {sB0, sB1};

    // Shared flag for stage signaling between warpgroups
    __shared__ volatile int stage_ready[2];

    int tid = threadIdx.x;
    int wg  = tid >> 7;        // 0 or 1
    int lane128 = tid & 127;   // lane within warpgroup
    int W   = lane128 / 32;
    int L   = lane128 % 32;
    int col_tiles = seq_len / N_TILE;

    // Both warpgroups cooperate to load Q into sQ_full
    for (int c = 0; c < N_CHUNKS; c++) {
        for (int i = tid; i < K_CHUNK * M_TILE; i += NUM_THREADS) {
            int row = i / K_CHUNK;
            int col = i % K_CHUNK;
            uint32_t dst = swizzle_b128(col, row);
            sQ_full[c * (K_CHUNK * M_TILE) + dst / 2] = myQ[row * D_HEAD + c * K_CHUNK + col];
        }
    }

    if (tid == 0) { stage_ready[0] = 0; stage_ready[1] = 0; }
    __syncthreads();

    // Only WG0 (threads 0-127) does compute. WG1 (128-255) does loads.
    float row_max_0 = -1e30f;
    float row_max_1 = -1e30f;
    float row_sum_0 = 0.0f;
    float row_sum_1 = 0.0f;

    float o_acc[O_COL_TILES][32];
    for (int ot = 0; ot < O_COL_TILES; ot++)
        for (int i = 0; i < 32; i++)
            o_acc[ot][i] = 0.0f;

    for (int ct = 0; ct < col_tiles; ct++) {
        const half* myK_tile = myK + (long long)ct * N_TILE * D_HEAD;
        const half* myV_tile = myV + (long long)ct * N_TILE * D_HEAD;

        // ========== Phase 1: S = Q * K^T ==========
        float s_accum[32];
        if (wg == 0) {
            for (int i = 0; i < 32; i++) s_accum[i] = 0.0f;
        }

        for (int c = 0; c < N_CHUNKS; c++) {
            int stage = c & 1;

            // WG1: load K chunk into sB[stage]
            if (wg == 1) {
                stage_ready[stage] = 0;
                __threadfence_block();
                load_tile_swizzled(sB_buf[stage], myK_tile, N_TILE, D_HEAD, c, lane128, 128);
                __threadfence_block();
                stage_ready[stage] = 1;
            }

            // WG0: wait for stage, then compute
            if (wg == 0) {
                while (!stage_ready[stage]) { __nanosleep(32); }

                half* sQ_chunk = sQ_full + c * (K_CHUNK * M_TILE);

                asm volatile("wgmma.fence.sync.aligned;\n");
                uint64_t dA = make_desc(sQ_chunk);
                uint64_t dB = make_desc(sB_buf[stage]);
                wgmma_m64n64k16(s_accum, dA, dB);
                asm volatile("wgmma.commit_group.sync.aligned;\n");
                asm volatile("wgmma.wait_group.sync.aligned 0;\n");
            }
        }

        // Barrier before Phase 2 (both WGs must sync)
        __syncthreads();

        if (wg == 0) {
            float p_local[32];
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

            // Phase 3: O += P * V
            for (int ot = 0; ot < O_COL_TILES; ot++) {
                for (int kc = 0; kc < PV_K_CHUNKS; kc++) {
                    store_p_chunk_to_smem(sA0, p_local, kc, lane128);

                    // Load V into sB0 (WG0 does its own load here for simplicity)
                    for (int i = lane128; i < K_CHUNK * N_TILE; i += 128) {
                        int jl = i / K_CHUNK;
                        int kl = i % K_CHUNK;
                        int v_row = kc * K_CHUNK + kl;
                        int v_col = ot * N_TILE + jl;
                        sB0[swizzle_b128(kl, jl) / 2] = myV_tile[v_row * D_HEAD + v_col];
                    }
                    // Need WG0-only barrier here. Use wgmma.fence as implicit barrier.
                    asm volatile("wgmma.fence.sync.aligned;\n");
                    uint64_t dA2 = make_desc(sA0);
                    uint64_t dB2 = make_desc(sB0);
                    wgmma_m64n64k16(o_acc[ot], dA2, dB2);
                    asm volatile("wgmma.commit_group.sync.aligned;\n");
                    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
                }
            }
        }

        // Sync before next col_tile
        __syncthreads();
    }

    // Final: O /= row_sum (WG0 only)
    if (wg == 0) {
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

int main() {
    int seq_lens[] = {512, 1024, 2048, 4096};
    int num_seq = 4;
    int heads = 32;
    int batch = 1;
    int warmup = 10;
    int runs = 100;

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     GAE - 2 Warpgroups: WG0=Compute WG1=Producer       ║\n");
    printf("║     Double-Buffered Stages - Async Load Overlap         ║\n");
    printf("║     Full S^2 - Online Softmax - Hopper SM90a            ║\n");
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

        dim3 grid(row_tiles, heads, batch);
        dim3 block(NUM_THREADS);

        for (int i = 0; i < warmup; i++)
            gae_multi_kernel<<<grid, block, SMEM_TOTAL>>>(dQ, dK, dV, dO, S, heads);
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
            gae_multi_kernel<<<grid, block, SMEM_TOTAL>>>(dQ, dK, dV, dO, S, heads);
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

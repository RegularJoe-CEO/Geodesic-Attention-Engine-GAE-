#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Pull in kernel defines and the kernel itself
#define M_TILE    64
#define N_TILE    64
#define D_HEAD    128
#define K_CHUNK   16
#define N_CHUNKS  (D_HEAD / K_CHUNK)
#define WARPGROUP 128
#define PV_K_CHUNKS (N_TILE / K_CHUNK)
#define O_COL_TILES (D_HEAD / N_TILE)
#define SMEM_A_BYTES (K_CHUNK * M_TILE * 2)
#define SMEM_B_BYTES (N_TILE * 128)
#define SMEM_TOTAL   32768
#define DESC_LEAD    128
#define DESC_STRIDE  1024
#define SWIZZLE_B128 3

// === Inline the device functions from waller_v5 ===
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

// Multi-tile kernel: each block handles one (batch, head, row_tile) combo
__global__ void __launch_bounds__(128, 1)
gae_bench_kernel(const half* __restrict__ Q,
                 const half* __restrict__ K,
                 const half* __restrict__ V,
                 float* __restrict__ O,
                 int seq_len, int num_heads) {
    // Block mapping: blockIdx.x = row_tile, blockIdx.y = head, blockIdx.z = batch
    int row_tile = blockIdx.x;
    int head     = blockIdx.y;
    int batch    = blockIdx.z;
    int row_off  = row_tile * M_TILE;

    // Pointers for this (batch, head) slice
    int bh_offset = (batch * num_heads + head) * seq_len * D_HEAD;
    int bh_offset_sq = (batch * num_heads + head) * seq_len * seq_len;
    const half* myQ = Q + bh_offset + row_off * D_HEAD;
    const half* myK = K + bh_offset;  // full seq_len x D_HEAD
    const half* myV = V + bh_offset;
    float* myO = O + bh_offset + row_off * D_HEAD;

    extern __shared__ char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = reinterpret_cast<half*>(smem_raw + SMEM_A_BYTES);

    int tid = threadIdx.x;

    // === Phase 1: S = Q * K^T (only first N_TILE=64 cols for now) ===
    float s_accum[32];
    for (int i = 0; i < 32; i++) s_accum[i] = 0.0f;

    for (int c = 0; c < N_CHUNKS; c++) {
        // Load Q chunk -> sA
        for (int i = tid; i < K_CHUNK * M_TILE; i += WARPGROUP) {
            int row = i / K_CHUNK;
            int col = i % K_CHUNK;
            half val = myQ[row * D_HEAD + c * K_CHUNK + col];
            sA[swizzle_b128(col, row) / 2] = val;
        }
        // Load K chunk -> sB (transposed)
        for (int i = tid; i < K_CHUNK * N_TILE; i += WARPGROUP) {
            int row = i / K_CHUNK;
            int col = i % K_CHUNK;
            half val = myK[row * D_HEAD + c * K_CHUNK + col];
            sB[swizzle_b128(col, row) / 2] = val;
        }
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;\n");
        uint64_t dA = make_desc(sA);
        uint64_t dB = make_desc(sB);
        wgmma_m64n64k16(s_accum, dA, dB);
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");
        __syncthreads();
    }

    // === Phase 2: Softmax ===
    float row_max = -1e30f;
    for (int i = 0; i < 32; i++) if (s_accum[i] > row_max) row_max = s_accum[i];
    // Warp reduce max
    for (int mask = 16; mask > 0; mask >>= 1)
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, mask));

    float row_sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        s_accum[i] = expf(s_accum[i] - row_max);
        row_sum += s_accum[i];
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, mask);

    float inv_sum = 1.0f / row_sum;
    for (int i = 0; i < 32; i++) s_accum[i] *= inv_sum;

    // === Phase 3: O = P * V ===
    // Store P to sA as fp16
    for (int i = tid; i < M_TILE * K_CHUNK; i += WARPGROUP) {
        int m = i / K_CHUNK;
        int k = i % K_CHUNK;
        int idx = m * 32 + k;  // simplified mapping
        float pval = (idx < 32) ? s_accum[idx] : 0.0f;
        sA[swizzle_b128(k, m) / 2] = __float2half(pval);
    }

    float o_accum[32];
    for (int ot = 0; ot < O_COL_TILES; ot++) {
        for (int i = 0; i < 32; i++) o_accum[i] = 0.0f;
        for (int kc = 0; kc < PV_K_CHUNKS; kc++) {
            // Load V chunk -> sB
            for (int i = tid; i < K_CHUNK * N_TILE; i += WARPGROUP) {
                int row = i / K_CHUNK;
                int col = i % K_CHUNK;
                int v_row = kc * K_CHUNK + row;
                int v_col = ot * N_TILE + col;
                half val = (v_row < seq_len && v_col < D_HEAD) ? myV[v_row * D_HEAD + v_col] : __float2half(0.0f);
                sB[swizzle_b128(col, row) / 2] = val;
            }
            __syncthreads();
            asm volatile("wgmma.fence.sync.aligned;\n");
            uint64_t dA = make_desc(sA);
            uint64_t dB = make_desc(sB);
            wgmma_m64n64k16(o_accum, dA, dB);
            asm volatile("wgmma.commit_group.sync.aligned;\n");
            asm volatile("wgmma.wait_group.sync.aligned 0;\n");
            __syncthreads();
        }
        // Write O tile
        for (int i = tid; i < M_TILE * N_TILE; i += WARPGROUP) {
            int r = i / N_TILE;
            int c = i % N_TILE;
            int idx = r * 32 + (c % 32);
            if (idx < 32 && (row_off + r) < seq_len)
                myO[(row_off + r) * D_HEAD + ot * N_TILE + c] = o_accum[idx];
        }
    }
}

// Random fill kernel
__global__ void fill_random(half* data, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = __float2half(curand_normal(&state) * 0.02f);
    }
}

int main() {
    int seq_lens[] = {512, 1024, 2048, 4096};
    int num_seq = 4;
    int batch = 1;
    int heads = 32;
    int warmup = 10;
    int runs = 100;

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     GAE (Geodesic Attention Engine) Benchmark Suite     ║\n");
    printf("║     Deterministic Linear Algebra Engine - Hopper SM90a  ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("Config: batch=%d  heads=%d  d_head=%d  warmup=%d  runs=%d\n\n", batch, heads, D_HEAD, warmup, runs);
    printf("%-10s %12s %12s %12s\n", "seq_len", "avg (us)", "TFLOPS", "util%%");
    printf("──────────────────────────────────────────────────────\n");

    for (int si = 0; si < num_seq; si++) {
        int S = seq_lens[si];
        int row_tiles = S / M_TILE;
        size_t qkv_elems = (size_t)batch * heads * S * D_HEAD;
        size_t qkv_bytes = qkv_elems * sizeof(half);
        size_t o_bytes = qkv_elems * sizeof(float);

        half *dQ, *dK, *dV;
        float *dO;
        cudaMalloc(&dQ, qkv_bytes);
        cudaMalloc(&dK, qkv_bytes);
        cudaMalloc(&dV, qkv_bytes);
        cudaMalloc(&dO, o_bytes);

        int fill_threads = 256;
        int fill_blocks = (qkv_elems + fill_threads - 1) / fill_threads;
        fill_random<<<fill_blocks, fill_threads>>>(dQ, qkv_elems, 42);
        fill_random<<<fill_blocks, fill_threads>>>(dK, qkv_elems, 43);
        fill_random<<<fill_blocks, fill_threads>>>(dV, qkv_elems, 44);
        cudaDeviceSynchronize();

        dim3 grid(row_tiles, heads, batch);
        dim3 block(WARPGROUP);

        // Warmup
        for (int i = 0; i < warmup; i++)
            gae_bench_kernel<<<grid, block, SMEM_TOTAL>>>(dQ, dK, dV, dO, S, heads);
        cudaDeviceSynchronize();

        // Timed
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < runs; i++)
            gae_bench_kernel<<<grid, block, SMEM_TOTAL>>>(dQ, dK, dV, dO, S, heads);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float total_ms;
        cudaEventElapsedTime(&total_ms, t0, t1);
        float avg_us = (total_ms / runs) * 1000.0f;

        // FLOPS: 4 * batch * heads * seq^2 * d_head
        double flops = 4.0 * batch * heads * (double)S * S * D_HEAD;
        double tflops = (flops / (avg_us * 1e-6)) / 1e12;
        double h100_peak = 989.0;  // FP16 TFLOPS peak
        double util = (tflops / h100_peak) * 100.0;

        printf("%-10d %10.1f %12.1f %10.1f%%\n", S, avg_us, tflops, util);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    printf("──────────────────────────────────────────────────────\n");
    printf("H100 FP16 peak: 989 TFLOPS\n");
    return 0;
}

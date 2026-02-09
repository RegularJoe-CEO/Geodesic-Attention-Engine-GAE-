#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ===== TEST MODE =====
// 1 = Identity-P + column-tag V (diagnostic)
// 0 = Real end-to-end QKV -> O (production)
#define DIAGNOSTIC_MODE 0


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
void wgmma_m64n64k16_transb0(float acc[32], uint64_t dA, uint64_t dB) {
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,1,1,1,0,0;\n"
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
void wgmma_fence()  {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait()   {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__
void reg_to_rowcol(int r, int W, int L, int &row, int &col) {
    row = 16 * W + (L >> 2) + 8 * ((r >> 1) & 1);
    col = (L & 3) * 2 + (r >> 2) * 8 + (r & 1);
}

__device__ __forceinline__
void store_p_chunk_to_smem(half* tile, float acc[32], int kc, int W, int L) {
    for (int r = 0; r < 32; r++) {
        int m, n;
        reg_to_rowcol(r, W, L, m, n);
        int j = n - kc * K_CHUNK;
        if (j >= 0 && j < K_CHUNK) {
            uint32_t sw = swizzle_b128(j, m);
            *reinterpret_cast<half*>(reinterpret_cast<char*>(tile) + sw) =
                __float2half(acc[r]);
        }
    }
}

__device__ __forceinline__
void online_softmax(float acc[32]) {
    unsigned mask = 0xFFFFFFFF;
    for (int sel = 0; sel < 2; sel++) {
        float mx = -1e30f;
        for (int r = 0; r < 32; r++)
            if (((r >> 1) & 1) == sel) mx = fmaxf(mx, acc[r]);
        mx = fmaxf(mx, __shfl_xor_sync(mask, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(mask, mx, 2));
        float s = 0.0f;
        for (int r = 0; r < 32; r++)
            if (((r >> 1) & 1) == sel) { acc[r] = expf(acc[r] - mx); s += acc[r]; }
        s += __shfl_xor_sync(mask, s, 1);
        s += __shfl_xor_sync(mask, s, 2);
        float inv = 1.0f / s;
        for (int r = 0; r < 32; r++)
            if (((r >> 1) & 1) == sel) acc[r] *= inv;
    }
}

__global__ void __launch_bounds__(128, 1)
waller_v5_kernel(const half* __restrict__ Q,
                 const half* __restrict__ K,
                 const half* __restrict__ V,
                 float* __restrict__ O,
                 float* __restrict__ P_dump,
                 int N_dim) {
    extern __shared__ char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = reinterpret_cast<half*>(smem_raw + SMEM_A_BYTES);

    int tid = threadIdx.x;
    int W   = tid / 32;
    int L   = tid % 32;

    /* Phase 1: S = Q * K^T */
    float s_accum[32];
    for (int i = 0; i < 32; i++) s_accum[i] = 0.0f;

    for (int c = 0; c < N_CHUNKS; c++) {
        for (int i = tid; i < K_CHUNK * M_TILE; i += WARPGROUP) {
            int dk = i / M_TILE, m = i % M_TILE;
            *reinterpret_cast<half*>(reinterpret_cast<char*>(sA) + swizzle_b128(dk, m)) =
                Q[m * D_HEAD + c * K_CHUNK + dk];
        }
        for (int i = tid; i < K_CHUNK * N_TILE; i += WARPGROUP) {
            int dk = i / N_TILE, n = i % N_TILE;
            *reinterpret_cast<half*>(reinterpret_cast<char*>(sB) + swizzle_b128(dk, n)) =
                K[n * D_HEAD + c * K_CHUNK + dk];
        }
        __syncthreads();
        wgmma_fence();
        if (tid==0) printf("DBG descA1=%016llx descB1=%016llx\n", (unsigned long long)make_desc(sA), (unsigned long long)make_desc(sB));

        wgmma_m64n64k16(s_accum, make_desc(sA), make_desc(sB));
        wgmma_commit();
        wgmma_wait();
        __syncthreads();
    }

    /* Phase 2: softmax */
    online_softmax(s_accum);

    /* Dump P to global for host verification */
    if (P_dump) {
        for (int r = 0; r < 32; r++) {
            int row, col;
            reg_to_rowcol(r, W, L, row, col);
            P_dump[row * N_TILE + col] = s_accum[r];
        }
    }

    /* Phase 3+4: O = P * V */
    for (int ot = 0; ot < O_COL_TILES; ot++) {
        int ocol = ot * N_TILE;
        float o_accum[32];
        for (int i = 0; i < 32; i++) o_accum[i] = 0.0f;

        for (int kc = 0; kc < PV_K_CHUNKS; kc++) {
#if DIAGNOSTIC_MODE
            /* DIAGNOSTIC: cooperative load P=Identity into sA */
            for (int i = tid; i < K_CHUNK * M_TILE; i += WARPGROUP) {
                int dk = i / M_TILE, m = i % M_TILE;
                int global_k = kc * K_CHUNK + dk;
                half val = (global_k == m) ? __float2half(1.0f) : __float2half(0.0f);
                *reinterpret_cast<half*>(reinterpret_cast<char*>(sA) + swizzle_b128(dk, m)) = val;
            }
#else
            /* REAL: store softmax(QK) from registers into sA */
            store_p_chunk_to_smem(sA, s_accum, kc, W, L);
            __syncthreads();
#endif

if (tid == 0 && kc == 0 && ot == 0 && P_dump) {
    int mismatches = 0;
    for (int m = 0; m < M_TILE; m++) {
        for (int j = 0; j < K_CHUNK; j++) {
            half sA_val = *reinterpret_cast<half*>(reinterpret_cast<char*>(sA) + swizzle_b128(j, m));
            float p_val = P_dump[m * N_TILE + j];
            if (__half2float(sA_val) != __half2float(__float2half(p_val))) {
                if (mismatches < 5) printf("MISMATCH row=%d col=%d sA=%.6f P_fp16=%.6f\n", m, j, __half2float(sA_val), __half2float(__float2half(p_val)));
                mismatches++;
            }
        }
    }
    printf("sA vs P_dump kc=0: %d / %d mismatches\n", mismatches, M_TILE * K_CHUNK);
}
            for (int i = tid; i < K_CHUNK * N_TILE; i += WARPGROUP) {
                int jl = i / N_TILE, n = i % N_TILE;
                *reinterpret_cast<half*>(reinterpret_cast<char*>(sB) + swizzle_b128(jl, n)) =
                    V[(kc * K_CHUNK + jl) * D_HEAD + ocol + n];
            }

              if (tid==0 && kc==0 && ot==0) {
                  printf("DBG sB row(jl=0) n=0..7: ");
                  for (int nn=0; nn<8; nn++) {
                      half hv = *reinterpret_cast<half*>(reinterpret_cast<char*>(sB) + swizzle_b128(0, nn));
                      printf("%.1f ", __half2float(hv));
                  }
                  printf("\n");
              }
            if (tid==0) printf("DBG descA3=%016llx descB3=%016llx\n", (unsigned long long)make_desc(sA), (unsigned long long)make_desc(sB));

            __syncthreads();
            wgmma_fence();
            wgmma_m64n64k16(o_accum, make_desc(sA), make_desc(sB));
            wgmma_commit();
            wgmma_wait();
            __syncthreads();
        }

        for (int r = 0; r < 32; r++) {
            int row, col;
            reg_to_rowcol(r, W, L, row, col);
            O[row * N_dim + ocol + col] = o_accum[r];
        }
    }
}

int main() {
    const int M = M_TILE, N = N_TILE;
    printf("=== Waller v5 - DLAE Fused Attention ===\n");
    printf("M=%d  N=%d  D_HEAD=%d\n\n", M, N, D_HEAD);

    size_t szQK = M * D_HEAD, szV = N * D_HEAD, szO = M * D_HEAD, szP = M * N;

    float* hQ  = (float*)malloc(szQK * 4);
    float* hK  = (float*)malloc(szQK * 4);
    float* hV  = (float*)malloc(szV  * 4);

#if DIAGNOSTIC_MODE
    // === DIAGNOSTIC OVERRIDE: Column-tag V[k][d] = d ===
    for (int k = 0; k < N; k++)
        for (int d = 0; d < D_HEAD; d++)
            hV[k * D_HEAD + d] = (float)d;
    // === END DIAGNOSTIC OVERRIDE ===
#endif

    float* hOgpu = (float*)malloc(szO * 4);
    float* hPgpu = (float*)malloc(szP * 4);

    srand(42);
    for (int i = 0; i < (int)szQK; i++) hQ[i] = (rand()/(float)RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < (int)szQK; i++) hK[i] = (rand()/(float)RAND_MAX - 0.5f) * 0.1f;
// DISABLED:     for (int k = 0; k < N; k++) for (int d = 0; d < D_HEAD; d++) hV[k*D_HEAD + d] = (float)d;
#if DIAGNOSTIC_MODE
    /* DIAGNOSTIC: column-tag V */
    for (int k = 0; k < N; k++)
        for (int d = 0; d < D_HEAD; d++)
            hV[k * D_HEAD + d] = (float)d;
#endif

    /* Convert to FP16 (same as GPU inputs) */
    half* hQh = (half*)malloc(szQK * 2);
    half* hKh = (half*)malloc(szQK * 2);
    half* hVh = (half*)malloc(szV  * 2);
    // RESTORE: build hVh from hV (required for FP16 ref + GPU)
    for (int i = 0; i < (int)szV; i++) hVh[i] = __float2half(hV[i]);
    // DEBUG: hVh sanity (expect 0..7 and 120..127 for column-tag)
    printf("DEBUG hVh row0 d=0..7: ");
    for (int d = 0; d < 8; d++) printf("%.1f ", __half2float(hVh[d]));
    printf("\nDEBUG hVh row0 d=120..127: ");
    for (int d = 120; d < 128; d++) printf("%.1f ", __half2float(hVh[d]));
    printf("\n");
    // DEBUG: hVh sanity (expect 0..7 and 120..127 in half)\

    printf("DEBUG hVh row0 d=0..7: ");\

    for (int d = 0; d < 8; d++) printf("%.1f ", __half2float(hVh[0 * D_HEAD + d]));\

    printf("\nDEBUG hVh row0 d=120..127: ");\

    for (int d = 120; d < 128; d++) printf("%.1f ", __half2float(hVh[0 * D_HEAD + d]));\

    printf("\n");\

    for (int i = 0; i < (int)szQK; i++) hQh[i] = __float2half(hQ[i]);
    for (int i = 0; i < (int)szQK; i++) hKh[i] = __float2half(hK[i]);
// DISABLED:     for (int k = 0; k < N; k++) for (int d = 0; d < D_HEAD; d++) hV[k*D_HEAD + d] = (float)d;

    /* ---- REF 1: Pure FP32 ---- */
    float* S32   = (float*)calloc(szP, 4);
    float* P32   = (float*)calloc(szP, 4);
    float* Oref32= (float*)calloc(szO, 4);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int d = 0; d < D_HEAD; d++) s += hQ[i*D_HEAD+d] * hK[j*D_HEAD+d];
            S32[i*N+j] = s;
        }
    for (int i = 0; i < M; i++) {
        float mx = -1e30f;
        for (int j = 0; j < N; j++) mx = fmaxf(mx, S32[i*N+j]);
        float s = 0;
        for (int j = 0; j < N; j++) { P32[i*N+j] = expf(S32[i*N+j]-mx); s += P32[i*N+j]; }
        for (int j = 0; j < N; j++) P32[i*N+j] /= s;
    }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D_HEAD; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += P32[i*N+k] * hV[k*D_HEAD+j];
            Oref32[i*D_HEAD+j] = s;
        }

    /* ---- REF 2: FP16-accurate (same precision path as GPU) ---- */
    float* S16   = (float*)calloc(szP, 4);
    float* P16   = (float*)calloc(szP, 4);
    float* Oref16= (float*)calloc(szO, 4);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int d = 0; d < D_HEAD; d++)
                s += __half2float(hQh[i*D_HEAD+d]) * __half2float(hKh[j*D_HEAD+d]);
            S16[i*N+j] = s;
        }
    for (int i = 0; i < M; i++) {
        float mx = -1e30f;
        for (int j = 0; j < N; j++) mx = fmaxf(mx, S16[i*N+j]);
        float s = 0;
        for (int j = 0; j < N; j++) { P16[i*N+j] = expf(S16[i*N+j]-mx); s += P16[i*N+j]; }
        for (int j = 0; j < N; j++) P16[i*N+j] /= s;
    }

    /* O16 = fp16(P16) * V_fp16, accumulated in FP32 — exact GPU path */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D_HEAD; j++) {
            float s = 0;
            for (int k = 0; k < N; k++)
                s += __half2float(__float2half(P16[i*N+k])) *
                     __half2float(hVh[k*D_HEAD+j]);
            Oref16[i*D_HEAD+j] = s;
        }

    printf("FP32 ref: S[0][0]=%.6f  P[0][0]=%.6f  O[0][0]=%.6f\n",
           S32[0], P32[0], Oref32[0]);
    printf("FP16 ref: S[0][0]=%.6f  P[0][0]=%.6f  O[0][0]=%.6f\n\n",
           S16[0], P16[0], Oref16[0]);
    printf("DEBUG Oref16 row0 cols 0-7: ");
    for (int d = 0; d < 8; d++) printf("%.2f ", (double)Oref16[d]);
    printf("\n");


    /* GPU */
    half *dQ, *dK, *dV; float *dO, *dP;
    cudaMalloc(&dQ, szQK*2);  cudaMalloc(&dK, szQK*2);
    cudaMalloc(&dV, szV*2);   cudaMalloc(&dO, szO*4);
    cudaMalloc(&dP, szP*4);
    cudaMemcpy(dQ, hQh, szQK*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hKh, szQK*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hVh, szV*2,  cudaMemcpyHostToDevice);

    printf("Launching kernel...\n");
    waller_v5_kernel<<<1, WARPGROUP, SMEM_TOTAL>>>(dQ, dK, dV, dO, dP, D_HEAD);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
    printf("Kernel OK\n\n");

    cudaMemcpy(hOgpu, dO, szO*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hPgpu, dP, szP*4, cudaMemcpyDeviceToHost);

    /* Verify P */
    float p_maxerr = 0;
    for (int i = 0; i < M*N; i++) {
        float d = fabsf(hPgpu[i] - P16[i]);
        if (d > p_maxerr) p_maxerr = d;
    }
    float psum0 = 0, psum63 = 0;
    for (int j = 0; j < N; j++) { psum0 += hPgpu[j]; psum63 += hPgpu[63*N+j]; }
    printf("GPU P: row0 sum=%.6f  row63 sum=%.6f\n", psum0, psum63);
    printf("GPU P[0][0]=%.6f  FP16ref P[0][0]=%.6f  P max err vs FP16ref: %.2e\n\n",
           hPgpu[0], P16[0], p_maxerr);

    /* Compare O vs both references */
    printf("GPU O row0 ALL 64 cols tile0: ");
    for(int d=0;d<64;d++) { if(d%8==0) printf("\n  [%d]: ",d); printf("%.0f ",hOgpu[d]); }
    printf("\n");
    printf("GPU O row0 ALL 64 cols tile1: ");
    for(int d=64;d<128;d++) { if(d%8==0) printf("\n  [%d]: ",d); printf("%.0f ",hOgpu[d]); }
    printf("\n");
    printf("GPU O row0 cols 0-7:  "); for(int d=0;d<8;d++) printf("%.2f ",hOgpu[d]); printf("\n");
    printf("GPU O row0 cols 60-67: "); for(int d=60;d<68;d++) printf("%.2f ",hOgpu[d]); printf("\n");
    printf("GPU O row0 cols 120-127: "); for(int d=120;d<128;d++) printf("%.2f ",hOgpu[d]); printf("\n");
    printf("REF O row0 cols 0-7:  "); for(int d=0;d<8;d++) printf("%.2f ",Oref32[d]); printf("\n");
    printf("GPU O[0][0]=%.6f  O[0][1]=%.6f\n\n", hOgpu[0], hOgpu[1]);

    for (int ref_id = 0; ref_id < 2; ref_id++) {
        float* ref = ref_id == 0 ? Oref32 : Oref16;
        const char* name = ref_id == 0 ? "FP32 ref" : "FP16 ref";
        float mx_e = 0, sum_e = 0; int bad = 0;
        for (int i = 0; i < M*D_HEAD; i++) {
            float d = fabsf(hOgpu[i] - ref[i]);
            float r = (fabsf(ref[i]) > 1e-6f) ? d/fabsf(ref[i]) : d;
            sum_e += d; if (d > mx_e) mx_e = d;
            if (r > 0.05f) bad++;
        }
        printf("vs %s:  max_err=%.6f  avg_err=%.6f  bad=%d/%d",
               name, mx_e, sum_e/(M*D_HEAD), bad, M*D_HEAD);
        if (bad == 0) printf("  *** PASS ***");
        printf("\n");
        if (bad > 0 && bad < 10) {
            for (int i = 0; i < M*D_HEAD; i++) {
                float d = fabsf(hOgpu[i] - ref[i]);
                float r = (fabsf(ref[i]) > 1e-6f) ? d/fabsf(ref[i]) : d;
                if (r > 0.05f)
                    printf("  O[%d][%d]: gpu=%.6f ref=%.6f diff=%.6f\n",
                           i/D_HEAD, i%D_HEAD, hOgpu[i], ref[i], d);
            }
        }
    }

    /* Timing */
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        waller_v5_kernel<<<1, WARPGROUP, SMEM_TOTAL>>>(dQ, dK, dV, dO, nullptr, D_HEAD);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    printf("\nAvg kernel time: %.2f us (100 iters)\n", ms * 10.0f);

    free(hQ); free(hK); free(hV); free(hOgpu); free(hPgpu);
    free(hQh); free(hKh); free(hVh);
    free(S32); free(P32); free(Oref32);
    free(S16); free(P16); free(Oref16);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dP);
    return 0;
}

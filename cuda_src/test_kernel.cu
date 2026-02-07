#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

#define TILE_SIZE 64
#define BLOCK_SIZE 256

__global__ void simple_test(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Do some actual work
        float val = 0.0f;
        for (int i = 0; i < 1000; i++) {
            val += sinf((float)idx + i);
        }
        out[idx] = val;
    }
}

int main() {
    int n = 1024 * 1024;
    float *d_out;
    
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    simple_test<<<n/256, 256>>>(d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        simple_test<<<n/256, 256>>>(d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time: %.3f ms (should be >0)\n", ms);
    
    CHECK_CUDA(cudaGetLastError());
    printf("No CUDA errors detected\n");
    
    cudaFree(d_out);
    return 0;
}

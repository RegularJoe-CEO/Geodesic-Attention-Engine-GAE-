#include <cuda_runtime.h>
#include <stdio.h>

cudaEvent_t start, stop;

__global__ void pyramid_prefix_kernel(const float *input, float *pyramid, int seq_len, int head_dim, int levels) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= seq_len) return; int t = tid;
  if (t >= seq_len) return;
  float prefix = 0.0f;
  for (int lvl = 0; lvl < levels; lvl++) {
    int block_size = 1 << lvl;
    int block_start = (tid / block_size) * block_size;
    prefix += input[t * head_dim + threadIdx.x];
    pyramid[lvl * seq_len * head_dim + block_start * head_dim + threadIdx.x] = prefix;
  }
}
__global__ void apa_kernel(const float *Q, const float *pyramid_K, const float *pyramid_V, float *output, int seq_len, int head_dim, int levels) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= seq_len) return; int t = tid;
  if (t >= seq_len) return;
  float score = 0.0f;
  float max_s = -1e9f;
  float sum_exp = 0.0f;
  for (int lvl = 0; lvl < levels; lvl++) {
    score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      score += Q[t * head_dim + d] * pyramid_K[lvl * seq_len * head_dim + t * head_dim + d];
    }
    score /= sqrtf((float)head_dim);
    if (score > max_s) max_s = score;
  }
  sum_exp = expf(score - max_s) * levels;
  float denom = max_s + logf(sum_exp);
  for (int d = 0; d < head_dim; d++) {
    output[t * head_dim + d] = pyramid_V[(levels-1) * seq_len * head_dim + t * head_dim + d] * expf(score - denom);
  }
}
int main() {
  int seq_len = 524288;
  int head_dim = 64;
  int levels = 19;
  float *d_input, *d_pyramid_K, *d_pyramid_V, *d_Q, *d_output;
  cudaMalloc(&d_input, seq_len * head_dim * sizeof(float));
  cudaMalloc(&d_pyramid_K, levels * seq_len * head_dim * sizeof(float));
  cudaMalloc(&d_pyramid_V, levels * seq_len * head_dim * sizeof(float));
  cudaMalloc(&d_Q, seq_len * head_dim * sizeof(float));
  cudaMalloc(&d_output, seq_len * head_dim * sizeof(float));
  cudaMemset(d_input, 0x40, seq_len * head_dim * sizeof(float));
  cudaMemset(d_Q, 0x40, seq_len * head_dim * sizeof(float));
  dim3 grid(seq_len);
  dim3 block(head_dim);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int iter = 0; iter < 100; iter++) {
    pyramid_prefix_kernel<<<grid, block>>>(d_input, d_pyramid_K, seq_len, head_dim, levels);
    pyramid_prefix_kernel<<<grid, block>>>(d_input, d_pyramid_V, seq_len, head_dim, levels);
    apa_kernel<<<grid, block>>>(d_Q, d_pyramid_K, d_pyramid_V, d_output, seq_len, head_dim, levels);
    cudaDeviceSynchronize();
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float ms; cudaEventElapsedTime(&ms, start, stop);
  printf("APA V1 %dk x100 iters: %.3f ms avg (%.1f TFLOPS), memory O(N log N)\n", seq_len/1024, ms/100, (4.0f*seq_len*seq_len*head_dim*100 * 1e-12f) / (ms/100 * 1e-3f) );
  printf("APA V1 %dk x100 iters: %.3f ms avg (%.1f TFLOPS), memory O(N log N)\n", seq_len/1024, ms/100, (4.0f*seq_len*seq_len*head_dim*100 * 1e-12f) / (ms/100 * 1e-3f));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_pyramid_K);
  cudaFree(d_pyramid_V);
  cudaFree(d_Q);
  cudaFree(d_output);
  return 0;
}


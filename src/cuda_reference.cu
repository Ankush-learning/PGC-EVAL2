// cuda_reference.cu
// Equivalent CUDA C++ reference implementation for nvcc environments.
// Compile: nvcc -O3 cuda_reference.cu -o gpu_mem_bench

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void ai_fused_activation(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    float alpha, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = alpha * x[i] + beta;
        y[i] = val > 0.0f ? val : 0.0f;  // Fused Affine + ReLU
    }
}

int main() {
    const int n = 16 * 1000 * 1000;
    const size_t nbytes = (size_t)n * sizeof(float);
    const float alpha = 1.75f, beta = -0.25f;

    float *h_x = nullptr, *h_y = nullptr;
    cudaMallocHost(&h_x, nbytes);  // Pinned host memory
    cudaMallocHost(&h_y, nbytes);
    for (int i = 0; i < n; ++i) h_x[i] = (float)(i % 100) * 0.01f - 0.5f;

    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, nbytes);
    cudaMalloc(&d_y, nbytes);

    cudaEvent_t ev_s, ev_h2d, ev_k, ev_d2h;
    cudaEventCreate(&ev_s);
    cudaEventCreate(&ev_h2d);
    cudaEventCreate(&ev_k);
    cudaEventCreate(&ev_d2h);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEventRecord(ev_s);
    cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
    cudaEventRecord(ev_h2d);
    ai_fused_activation<<<blocks, threads>>>(d_x, d_y, alpha, beta, n);
    cudaEventRecord(ev_k);
    cudaMemcpy(h_y, d_y, nbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(ev_d2h);
    cudaEventSynchronize(ev_d2h);

    float h2d_ms = 0, k_ms = 0, d2h_ms = 0;
    cudaEventElapsedTime(&h2d_ms, ev_s, ev_h2d);
    cudaEventElapsedTime(&k_ms, ev_h2d, ev_k);
    cudaEventElapsedTime(&d2h_ms, ev_k, ev_d2h);

    printf("Pinned Explicit (16M floats): H2D=%.3f ms, Kernel=%.3f ms, D2H=%.3f ms, Total=%.3f ms\n",
           h2d_ms, k_ms, d2h_ms, h2d_ms + k_ms + d2h_ms);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    return 0;
}

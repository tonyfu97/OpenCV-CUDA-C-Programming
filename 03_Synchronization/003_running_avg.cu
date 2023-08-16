#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#define N 10000000

// Kernel to compute the running average on GPU
__global__ void runningAverageGPU(const float* d_data, float* d_result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sharedSum[1024];

    sharedSum[threadIdx.x] = (tid < N) ? d_data[tid] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = 512; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Store the result
    if (threadIdx.x == 0) {
        atomicAdd(d_result, sharedSum[0]);
    }
}

// Function to compute the running average on CPU
void runningAverageCPU(const float* h_data, float* result) {
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += h_data[i];
    }
    *result = sum / N;
}

int main() {
    float* h_data = new float[N];
    float cpu_result = 0, gpu_result = 0;
    float* d_data, * d_result;

    // Initialization
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(i);
    }

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    // Timing for CPU
    auto start = std::chrono::high_resolution_clock::now();
    runningAverageCPU(h_data, &cpu_result);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time for CPU: " << duration.count() << " microseconds\n";

    // Timing for GPU
    start = std::chrono::high_resolution_clock::now();
    runningAverageGPU << <(N + 1023) / 1024, 1024 >> > (d_data, d_result);
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time for GPU: " << duration.count() << " microseconds\n";

    cudaMemcpy(&gpu_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    gpu_result /= N;

    std::cout << "CPU Result: " << cpu_result << "\nGPU Result: " << gpu_result << "\n";

    // Cleanup
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}

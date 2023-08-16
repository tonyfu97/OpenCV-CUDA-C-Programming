#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N 1000000

// Device code
__global__ void addGlobal(int* a, int* b, int* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) c[tid] = a[tid] + b[tid];
}

__global__ void addShared(int* a, int* b, int* c) {
    __shared__ int sharedA[1024], sharedB[1024];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    sharedA[threadIdx.x] = a[tid];
    sharedB[threadIdx.x] = b[tid];
    __syncthreads();

    if (tid < N) c[tid] = sharedA[threadIdx.x] + sharedB[threadIdx.x];
}

__global__ void addLocal(int a, int b, int* c) {
    int tid = threadIdx.x;
    int localA = a;
    int localB = b;
    if (tid < N) c[tid] = localA + localB;
}

__constant__ int constA, constB;

__global__ void addConstant(int* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) c[tid] = constA + constB;
}

int main() {
    int* h_a, * h_b, * h_c, * d_a, * d_b, * d_c;

    // Allocation and initialization
    h_a = new int[N];
    h_b = new int[N];
    h_c = new int[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = 1;
        h_b[i] = 2;
    }
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Timing for global memory
    auto start = std::chrono::high_resolution_clock::now();
    addGlobal << <(N + 255) / 256, 256 >> > (d_a, d_b, d_c);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time for global memory: " << duration.count() << " microseconds\n";

    // Timing for shared memory
    start = std::chrono::high_resolution_clock::now();
    addShared << <(N + 255) / 256, 256 >> > (d_a, d_b, d_c);
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time for shared memory: " << duration.count() << " microseconds\n";

    // Timing for local memory
    start = std::chrono::high_resolution_clock::now();
    addLocal << <(N + 255) / 256, 256 >> > (1, 2, d_c);
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time for local memory: " << duration.count() << " microseconds\n";

    // Timing for constant memory
    cudaMemcpyToSymbol(constA, h_a, sizeof(int));
    cudaMemcpyToSymbol(constB, h_b, sizeof(int));
    start = std::chrono::high_resolution_clock::now();
    addConstant << <(N + 255) / 256, 256 >> > (d_c);
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time for constant memory: " << duration.count() << " microseconds\n";

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

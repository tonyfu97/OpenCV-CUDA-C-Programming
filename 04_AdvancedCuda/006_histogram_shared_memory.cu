#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>

#define N 1000
#define NUM_BINS 256

__global__ void histogram_shared_memory(int* d_a, int* d_b)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int offset = blockDim.x * gridDim.x;

	__shared__ int cache[NUM_BINS];
	cache[threadIdx.x] = 0;
	__syncthreads();

	while (tid < N)
	{
		atomicAdd(&(cache[d_a[tid]]), 1);
		tid += offset;
	}
	__syncthreads();
	atomicAdd(&(d_b[threadIdx.x]), cache[threadIdx.x]);
}

int main()
{
	int* h_a = new int[N];
	for (int i = 0; i < N; i++)
	{
		h_a[i] = i % NUM_BINS;
	}

	int* h_b = new int[N];
	std::fill(h_b, h_b + N, 0);

	int* d_a, * d_b;
	cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_b, NUM_BINS * sizeof(int));

	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, NUM_BINS * sizeof(int), cudaMemcpyHostToDevice);

	histogram_shared_memory << <(N + 255) / NUM_BINS, NUM_BINS >> > (d_a, d_b);

	cudaMemcpy(h_b, d_b, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << "Histogram: \n";
	for (int i = 0; i < NUM_BINS; i++)
	{
		std::cout << "bin [" << i << "]: count: " << h_b[i] << "\n";
	}

	cudaFree(d_a);
	cudaFree(d_b);
	delete[] h_a;
	delete[] h_b;

	return 0;
}
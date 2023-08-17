#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define N 50000

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		d_c[tid] = d_a[tid] + d_b[tid];
		tid += blockDim.x + gridDim.x;
	}
}

int main()
{
	int* h_a, * h_b, * h_c;
	int* d_a0, * d_b0, * d_c0;
	int* d_a1, * d_b1, * d_c1;

	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	cudaEvent_t e_start, e_stop;
	cudaEventCreate(&e_start);
	cudaEventCreate(&e_stop);
	cudaEventRecord(e_start, 0);

	// Allocate pinned memory on the host
	cudaHostAlloc((void**)&h_a, N * 2 * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_b, N * 2 * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_c, N * 2 * sizeof(int), cudaHostAllocDefault);

	cudaMalloc((void**)&d_a0, N * sizeof(int));
	cudaMalloc((void**)&d_b0, N * sizeof(int));
	cudaMalloc((void**)&d_c0, N * sizeof(int));
	cudaMalloc((void**)&d_a1, N * sizeof(int));
	cudaMalloc((void**)&d_b1, N * sizeof(int));
	cudaMalloc((void**)&d_c1, N * sizeof(int));

	for (int i = 0; i < N * 2; i++)
	{
		h_a[i] = 2 * i * i;
		h_b[i] = i;
	}

	cudaMemcpyAsync(d_a0, h_a, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(d_a1, h_a + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_b0, h_b, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(d_b1, h_b + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

	gpuAdd << <512, 512, 0, stream0 >> > (d_a0, d_b0, d_c0);
	gpuAdd << <512, 512, 0, stream1 >> > (d_a1, d_b1, d_c1);

	cudaMemcpyAsync(h_c, d_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
	cudaMemcpyAsync(h_c + N, d_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);

	cudaDeviceSynchronize();
	cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
	printf("Time to add %d numbers: %3.1f ms\n", 2 * N, elapsedTime);

	int Correct = 1;
	printf("Vector addition on GPU \n");
	for (int i = 0; i < 2 * N; i++) {
		if ((h_a[i] + h_b[i] != h_c[i]))
		{
			Correct = 0;
		}
	}
	if (Correct == 1)
	{
		printf("GPU has computed Sum Correctly\n");
	}
	else
	{
		printf("There is an Error in GPU Computation\n");
	}

	cudaFree(d_a0);
	cudaFree(d_b0);
	cudaFree(d_c0);
	cudaFree(d_a1);
	cudaFree(d_b1);
	cudaFree(d_c1);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
	return 0;
}
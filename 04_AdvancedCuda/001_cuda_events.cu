#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 50000

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	while (tid < N)
	{
		d_c[tid] = d_a[tid] + d_b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main()
{
	int* h_a = new int[N];
	int* h_b = new int[N];
	int* h_c = new int[N];
	int* d_a, * d_b, * d_c;

	cudaEvent_t e_start, e_stop;
	cudaEventCreate(&e_start);
	cudaEventCreate(&e_stop);
	cudaEventRecord(e_start, 0);

	cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMalloc((void**)&d_c, N * sizeof(int));

	for (int i = 0; i < N; i++)
	{
		h_a[i] = 2 * i * i;
		h_b[i] = i;
	}

	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
	
	gpuAdd << <512, 512 >> > (d_a, d_b, d_c);

	cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
	printf("Time to add %d numbers: %3.1f ms\n", N, elapsedTime);

	int Correct = 1;
	printf("Vector addition on GPU \n");
	//Printing result on console
	for (int i = 0; i < N; i++) {
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

	delete[] h_a;
	delete[] h_b;
	delete[] h_c;
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define ARRAY_SIZE 5
#define THREAD_PER_BLOCK 5

__global__ void rankSort(int* d_a, int* d_b)
{
	int count = 0;
	int tid = threadIdx.x;
	int ttid = blockIdx.x * THREAD_PER_BLOCK + tid;
	int val = d_a[ttid];

	__shared__ int cache[THREAD_PER_BLOCK];
	for (int i = tid; i < ARRAY_SIZE; i+=THREAD_PER_BLOCK)
	{
		cache[tid] = d_a[i];
		__syncthreads();
		for (int j = 0; j < THREAD_PER_BLOCK; j++)
		{
			if (val > cache[j])
			{
				count++;
			}
			__syncthreads();
		}
	}
	d_b[count] = val;
}

int main()
{
	int h_a[ARRAY_SIZE] = { 5, 9, 3, 4, 8 };
	int h_b[ARRAY_SIZE];
	int* d_a, * d_b;

	cudaMalloc((void**)&d_a, ARRAY_SIZE * sizeof(int));
	cudaMalloc((void**)&d_b, ARRAY_SIZE * sizeof(int));
	
	cudaMemcpy(d_a, h_a, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	rankSort << <(ARRAY_SIZE + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>> (d_a, d_b);

	// cudaDeviceSynchronize(); This is not neccessary

	cudaMemcpy(h_b, d_b, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	printf("The sorted array is : \n");
	for (int i = 0; i < ARRAY_SIZE - 1; i++)
	{
		printf("%d, ", h_b[i]);
	}
	printf("%d\n", h_b[ARRAY_SIZE - 1]);

	cudaFree(d_a);
	cudaFree(d_b);
	return 0;
}
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		d_c[tid] = d_a[tid] + d_b[tid];
		tid += gridDim.x * blockDim.x;

		/*printf("gridDim.x = %d\n blockDim.x = %d\n", gridDim.x, blockDim.x);*/
	}
}

int main()
{
	int h_a[N], h_b[N], h_c[N];
	int* d_a, * d_b, * d_c;

	cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMalloc((void**)&d_c, N * sizeof(int));

	// Initialize the arrays
	for (int i = 0; i < N; i++)
	{
		h_a[i] = 2 * i * i;
		h_b[i] = i;
	}

	// Copy the input array from host to device
	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

	// Kernel call
	gpuAdd << <2, 3 >> > (d_a, d_b, d_c);

	// Copy the result from device to host
	cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Print the results
	printf("Vector addition on GPU\n");
	for (int i = 0; i < N; i++)
	{
		printf("%d-th element: %d\n", i, h_c[i]);
	}

	// Free GPU memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
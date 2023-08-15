#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


__global__ void gpuAdd(int d_a, int d_b, int* d_c)
{
	*d_c = d_a + d_b;
}

int main()
{
	int h_c; // Host variable to store answer
	int* d_c; // Device pointer

	cudaMalloc((void**)&d_c, sizeof(int));
	gpuAdd <<<1, 1 >>> (1, 4, d_c);

	cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("1 + 4 = %d/n", h_c);

	cudaFree(d_c);
	return 0;
}
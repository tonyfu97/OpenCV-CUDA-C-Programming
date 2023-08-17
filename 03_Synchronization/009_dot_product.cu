#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 512

__global__ void gpu_dot(float* d_a, float* d_b, float* d_c)
{
	__shared__ float partial_sum[THREADS_PER_BLOCK];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int index = threadIdx.x;

	float sum = 0;
	while (tid < N)
	{
		sum += d_a[tid] * d_b[tid];
		tid += blockDim.x * gridDim.x;
	}

	partial_sum[index] = sum;

	__syncthreads();

	int i = blockDim.x / 2;

	while (i != 0)
	{
		if (index < i)
		{
			partial_sum[index] += partial_sum[index + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (index == 0)
	{
		d_c[blockIdx.x] = partial_sum[0];
	}
}

int main()
{
	float *h_a, *h_b, h_c, *partial_sum;
	float *d_a, *d_b, *d_partial_sum;
	int block_calc = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int blocks_per_grid = (32 < block_calc) ? 32 : block_calc;

	h_a = (float*)malloc(N * sizeof(float));
	h_b = (float*)malloc(N * sizeof(float));
	partial_sum = (float*)malloc(blocks_per_grid * sizeof(float));

	cudaMalloc((void**)&d_a, N * sizeof(float));
	cudaMalloc((void**)&d_b, N * sizeof(float));
	cudaMalloc((void**)&d_partial_sum, blocks_per_grid * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		h_a[i] = i;
		h_b[i] = 2;
	}

	cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

	gpu_dot << <blocks_per_grid, THREADS_PER_BLOCK >> > (d_a, d_b, d_partial_sum);

	cudaMemcpy(partial_sum, d_partial_sum, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);

	// Calculate final dot product on host
	h_c = 0;
	for (int i = 0; i < blocks_per_grid; i++) {
		h_c += partial_sum[i];
	}
	printf("The computed dot product is: %f\n", h_c);
#define cpu_sum(x) (x*(x+1))
	if (h_c == cpu_sum((float)(N - 1)))
	{
		printf("The dot product computed by GPU is correct\n");
	}
	else
	{
		printf("Error in dot product computation");
	}

	// free memory on host and device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_partial_sum);
	free(h_a);
	free(h_b);
	free(partial_sum);
}
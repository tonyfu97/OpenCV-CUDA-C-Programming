# Chapter 03: Threads, Synchronization, and Memory - Learning Reflections

**Author**: Tony Fu  
**Date**: August 14, 2023  
**Hardware and Software Configurations**: See [README.md](../README.md) at the repo root

**Reference**: Chapter 3 of [*Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA*](https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA.git) by Bhaumik Vaidya.

## Core Concepts

### A Trick with Large N

When the problem size `N` is large, we might not have enough blocks and threads. In this case, we can use a while loop inside the kernel function to iterate over the next unprocessed data point. This continues until all data have been processed. For example:

```cpp
#define N 9999999
gpuAdd <<<512, 1024>>> (d_a, d_b, d_c);
```
Apparently, the `<<<blocks, threads>>>` means that we can at most process *512 * 1024 = 524,288* data points in parallel with the kernel we had in Chapter 2. We can access the quantity `blocks` with `gridDim.x`, and the quantity `threads` with `blockDim.x`.

```cpp
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		d_c[tid] = d_a[tid] + d_b[tid];
		tid += gridDim.x * blockDim.x;
	}
}
```





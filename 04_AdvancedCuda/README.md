# Chapter 04: Advanced Concepts in CUDA - Learning Reflections

**Author**: Tony Fu  
**Date**: August 16, 2023  
**Hardware and Software Configurations**: See [README.md](../README.md) at the repo root

**Reference**: Chapter 4 of [*Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA*](https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA.git) by Bhaumik Vaidya.

## Core Concepts

### 1. Getting Rid of Annoying Red Squiggly Lines under CUDA keywords

Many people have found that in Visual Studio, red squiggly lines appear under CUDA keywords such as `threadIdx`, `blockDim`, etc., even though the syntax is perfectly correct. Including the header:

```cpp
#include <device_launch_parameters.h>
```

may resolve this issue. However, it might not work in all cases, and the solution can depend on the specific version of Visual Studio and the project configuration.

### 2. CUDA Events

CUDA events can be used to measure the time elapsed between different points in the code. They are commonly used to profile both data transfer and kernel execution times.

1. **Creating CUDA Events**:
```cpp
cudaEvent_t e_start, e_stop;
cudaEventCreate(&e_start);
cudaEventCreate(&e_stop);
```
This pattern of declaring an object and using another function to initialize it is common in C, which doesn't have the concept of classes and constructors. It emulates object-oriented design without constructors, like in the case of `cudaMalloc`.

2. **Recording Events**:
```cpp
cudaEventRecord(event, stream);
```
In the [example](001_cuda_events.cu), the default stream is used by passing `0` as the `stream` argument. To ensure all device code is finished before recording the stop time, use:
```cpp
cudaDeviceSynchronize();
```
To wait for the event to finish recording the stop time, use:
```cpp
cudaEventSynchronize(e_stop);
```

3. **Calculating the Elapsed Time**:
```cpp
cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
printf("Time to add %d numbers: %3.1f ms\n", N, elapsedTime);
```

By wrapping GPU-related code with CUDA events, precise timing measurements can be obtained, including data transfer and kernel execution. If only interested in the kernel execution time, the events can be placed around the kernel call specifically.


### 3. Error Checking

1. **Utilizing cudaError_t**:
   Use `cudaError_t` to hold the return status and check if it was successful.
   ```cpp
   cudaError_t cudaStatus;
   cudaStatus = cudaMalloc((void**)&d_c, sizeof(int));
   if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "cudaMalloc failed!");
       goto Error;
   }
   ```

2. **Checking Memory Allocations and Copies**:
   Check after calls like `cudaMalloc` and `cudaMemcpy`.
   ```cpp
   cudaStatus = cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
   if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "cudaMemcpy failed!");
       goto Error;
   }
   ```

3. **Handling Kernel Launch Errors**:
   Check if the kernel launch was successful using `cudaGetLastError`.
   ```cpp
   gpuAdd << <1, 1 >> > (d_a, d_b, d_c);
   cudaStatus = cudaGetLastError();
   if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
       goto Error;
   }
   ```

4. **Using Error Messages**:
   Use `cudaGetErrorString` for more descriptive error messages.
   ```cpp
   fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
   ```

5. **Error Handling Strategy with `goto`**:
   Use `goto` for a common cleanup section.
   ```cpp
   if (cudaStatus != cudaSuccess) {
       goto Error;
   }
   // ...
   Error:
   cudaFree(d_c);
   cudaFree(d_a);
   cudaFree(d_b);
   ```

6. **Cleanup**:
   Free device memory with `cudaFree`.
   ```cpp
   cudaFree(d_c);
   cudaFree(d_a);
   cudaFree(d_b);
   ```

### 4. Rules of Thumb for Blocks and Threads in CUDA

**Number of Blocks**:
- **Multiple of Multiprocessors**: Aligning the number of blocks with a multiple of the GPU's multiprocessors can enhance scheduling efficiency. Accordingly to the book, a commonly suggested ratio is 2, meaning that having twice as many blocks as multiprocessors often yields good performance.
- **Occupancy**: Consider the occupancy, or the ratio of active warps to the maximum number of warps supported by the SM, to choose an appropriate number of blocks.

**Number of Threads**:
- **Multiples of Warp Size**: Choose thread blocks with sizes that are multiples of the warp size (typically 32 for NVIDIA GPUs) to ensure that there are no idle threads within a warp.
- **Balance and Experiment**: Start with common sizes like 128 or 256 threads per block, but experiment and profile to find the best balance between parallelism and the kernel's resource requirements.


### 5. CUDA Streams

Until now, we have launched only one kernel on the device at a time. What happens if we want to launch multiple kernels simultaneously? We can't use `cudaMemcpy()` for this purpose, as it's a "blocking" operation on the CPU. Instead, we need to create CUDA streams to launch separate kernels concurrently. This requires the use of asynchronous operations such as `cudaMemcpyAsync()` to copy memory, along with other asynchronous functions for CPU-centered tasks. But how can we ensure that the memory is properly copied before the kernel is launched? This leads us to the concept of CUDA streams.

CUDA streams are sequences of commands (including memory transfers, kernel launches, etc.) that execute in a specified order on the GPU. By default, all executions are done within the default stream (with index 0). However, creating separate streams allows different sequences to execute concurrently. Here's how you can create a stream:

```cpp
cudaStream_t stream0;
cudaStreamCreate(&stream0);
```

Don't forget to delete the stream at the end of the program:

```cpp
cudaStreamDestroy(stream0);
```

Though CUDA streams execute in order, there may still be times when we'd like to wait for them to complete. We can use `cudaStreamSynchronize(stream)` to wait for a specific stream to complete all its operations. Alternatively, we can call `cudaDeviceSynchronize()` to wait for all streams to complete. It's worth noting that using both at the same time, as seen in the example `03_cuda_stream.cu` from the book, is redundant.

### 6. Paralellized Sorting

Unlike the previous operations we seen, which are mainly map (one-to-one), transpose (one-to-one), scatter (one-to-many), gather (many to one), stencil(specialized gather, also many-to-one), reduce (all-to-one), operations, sorting is an all-to-all operations. 

Here's a table that summarizes some common sorting algorithms:

| Algorithm      | Space Complexity | Time Complexity      | GPU Acceleratable | Description                                                                                     |
|----------------|------------------|----------------------|-------------------|-------------------------------------------------------------------------------------------------|
| Quick Sort     | \(O(\log n)\)         | \(O(n \log n)\) (average) | Yes             | Partitions the array and recursively sorts the partitions.                                      |
| Merge Sort     | \(O(n)\)              | \(O(n \log n)\)         | Yes             | Recursively divides the array and then merges the sorted partitions.                             |
| Bubble Sort    | \(O(1)\)              | \(O(n^2)\)              | No              | Repeatedly swaps adjacent elements if they are in the wrong order.                               |
| Insertion Sort | \(O(1)\)              | \(O(n^2)\)              | No              | Builds a sorted array one element at a time.                                                     |
| Heap Sort      | \(O(1)\)              | \(O(n \log n)\)         | Yes             | Uses a binary heap to sort the elements.                                                         |
| Radix Sort     | \(O(nk)\)             | \(O(nk)\)               | Yes             | Sorts integers by processing individual digits.                                                  |
| Selection Sort | \(O(1)\)              | \(O(n^2)\)              | No              | Selects the minimum/maximum element and places it at the beginning/end of the list.               |
| Rank Sort      | \(O(n)\)              | \(O(n^2)\) to \(O(n \log n)\) depending on implementation | Yes, see example       | Determines the rank of each element and places it in the corresponding position in the sorted list. |

The book implements rank sort in [`004_rank_sort.cu`](004_rank_sort.cu). Here's how it's done:

```cpp
__global__ void addKernel(int* d_a, int* d_b)
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
```

 The code is attempting to determine the rank of each element in the array by counting how many numbers are less than it. This rank is then used to place the number in the corresponding position in the output array.

However, the author's code here did not account for repeating numbers with the same value, they will have the same rank, and the code will attempt to place them in the same position in the output array (`d_b`). This leads to a race condition where the last thread to write to that position will overwrite any previous writes. The result is undefined behavior, and some of the repeating numbers will be lost.

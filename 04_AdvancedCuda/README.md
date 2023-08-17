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




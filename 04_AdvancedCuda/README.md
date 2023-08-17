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


# Chapter 02: Parallel Programming - Learning Reflections

**Author**: Tony Fu  
**Date**: August 14, 2023  
**Hardware and Software Configurations**: See [README.md](../README.md) at the repo root

**Reference**: Chapter 2 of [*Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA*](https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA.git) by Bhaumik Vaidya.

## Core Concepts

### Hierarchy of GPU

- **Device**: The GPU itself
	- **Grid**: A collection of blocks.
		- **Block**: All threads in a block run on the same Stream Multiprocessor (SM).
			- **Warp**: A group of 32 threads within a block.
				- **Thread**: The smallest unit of execution. Runs the same kernel code but operates on different data.

### Basic Workflow

Here is the general workflow:

1. Include the necessary headers
```cpp
#include <cuda.h>
#include <cuda_runtime.h>
```

2. Define the kernel function
```cpp
__global__ void gpuAdd(int d_a, int d_b, int *d_c) {
	*d_c = d_a + d_b;
}
```
There are three qualifiers:

| Qualifier				| Called from Host | Executed on |
|-----------------------|------------------|-------------|
| __global__			| Yes              | Device      |
| __host__ (default)    | Yes              | Host        |
| __device__			| No               | Device      |

When the code compiles, the host compiler will ignore the `__device__` qualifier, and nvcc will ignore the `__host__` qualifier. The compilers will generate the appropriate code for each context, ensuring that it runs efficiently on the target (host or device).

On the other hand, the `__global__` qualifier is an unique case. The host compiler will generate code that sets up the kernel launch on the device. This code includes configuration like the grid and block dimensions. Then, the device code compiler will generate the actual kernel code that gets executed on the GPU. So, we can think of the `__global__` qualifier as creating two separate parts: one for launching the kernel (host side) and one for executing the kernel (device side).

3. Allocate memory for the device pointer
```cpp
int *d_c;  // device pointer to either an int or an array of int
cudaMalloc((void**)&d_c, sizeof(int));
```

By using a void pointer, the function is made general. It doesn't matter what specific type of data you're allocating space for on the device; you can use cudaMalloc to allocate that space. The user is responsible for casting to the correct type.

By passing a pointer to the pointer (i.e., a reference to the pointer), cudaMalloc is able to modify the value of the pointer itself. This allows it to set the pointer to the location of the newly allocated memory on the device.

In CUDA programming, host pointers point to memory locations on the CPU, and device pointers point to memory locations on the GPU. It is a good practice to prefix all device pointers with `d_` and avoid modifying them in the host code. On the other hand, it is rare and generally not advisable to have host pointers within the device code in CUDA, reducing the likelihood of accidental modification of host pointers within the device code.

4. Call the kernel function. This is referred to as a "kernel call" in the book
```cpp
gpuAdd <<<1, 1>>> (2, 6, d_c);
```
The syntax `<<<blocks, threads>>>` is a special syntax used by nvcc to specify that the kernel code should be executed on one block with one thread. We can optionally pass a third argument, which specifies the number of bytes available to each block. 



5. Copy the result from device to host
```cpp
int h_c;
cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
```

- **Destination Pointer (`&h_c`)**
- **Source Pointer (`d_c`)**
- **Size (`sizeof(int)`)**: This is the number of bytes to be copied. `sizeof(int)` specifies the size of an integer in bytes..
- **Direction (`cudaMemcpyDeviceToHost`)**: This parameter specifies the direction of the copy. There are also other directions: `cudaMemcpyHostToHost`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, and `cudaMemcpyDeviceToDevice`.


6. Free up memory
```cpp
cudaFree(d_c);
```
Free the memory on the device.



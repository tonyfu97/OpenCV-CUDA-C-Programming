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

### 3. Device Properties
`cuda_runtime.h` contains some built-in functions to access the device properties: 
```
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    std::cout << "Total CUDA devices found: " << deviceCount << std::endl;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "\nDevice " << device << " (" << deviceProp.name << ") properties:\n";
        std::cout << "  CUDA version: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max thread dimensions: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid dimensions: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Total constant memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
    }

    return 0;
}
```
Here is the output:
```
Total CUDA devices found: 1

Device 0 (NVIDIA GeForce RTX 3080) properties:
  CUDA version: 8.6
  Global memory: 10239 MB
  Shared memory per block: 48 KB
  Warp size: 32
  Max threads per block: 1024
  Max thread dimensions: (1024, 1024, 64)
  Max grid dimensions: (2147483647, 65535, 65535)
  Number of multiprocessors: 68
  Clock rate: 1710 MHz
  Total constant memory: 64 KB
```

### 4. Vector Operations

We can parallel across blocks like `gpuAdd << <N, 1 >> >(d_a, d_b, d_c);`
```
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
int tid = blockIdx.x;
if (tid < N)
	d_c[tid] = d_a[tid] + d_b[tid];
}
```

...or parallelize across threads like `gpuAdd << <1, N >> >(d_a, d_b, d_c);`
```
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
int tid = threadIdx.x;
if (tid < N)
	d_c[tid] = d_a[tid] + d_b[tid];
}
```
For large-scale problems, a combination of both blocks and threads (e.g., `gpuAdd <<<M, P>>>(d_a, d_b, d_c);` where `M` and `P` are the number of blocks and threads per block, respectively) is generally used to fully exploit the hierarchical parallelism provided by CUDA.

For small-scale problems or specific optimization purposes, using multiple blocks (first example) can allow for better utilization of the GPU's resources. Different blocks may be scheduled on different multiprocessors, allowing for true parallel execution. It provides more scalability, especially when N is large, as the number of blocks can be much larger than the number of threads per block. On the other hand, using multiple threads in a single block (second example) can be a very efficient way to organize the computation if N is small. However, there is a limitation to the number of threads per block (e.g., 1024 in many GPUs).

### 5. Communication Patterns

Here are some communication patterns, including some that weren't mentioned in the book (thanks, ChatGPT!):

Communication patterns describe how data is exchanged between different computational elements, such as processors, cores, or threads, within a parallel computing environment. Understanding these patterns is crucial for designing efficient parallel algorithms. Here are some common communication patterns along with their typical applications:

1. **Point-to-Point Communication**:
   - **Description**: Direct communication between two specific processing elements.
   - **Applications**: Basic data exchanges, client-server communication, and any scenario where precise control of data transfer is needed.

2. **Broadcast**:
   - **Description**: A single processing element sends the same data to all other processing elements.
   - **Applications**: Distributing constants or parameters to all processing elements, initializing shared configurations.

3. **Gather**:
   - **Description**: Collecting data from all processing elements to a single processing element.
   - **Applications**: Summarizing results, combining partial solutions, or preparing data for output.

4. **Scatter**:
   - **Description**: Distributing different pieces of data from one processing element to many others.
   - **Applications**: Allocating tasks or distributing initial data for parallel processing.

5. **Reduce**:
   - **Description**: Combining data from all processing elements using a specific operation (such as sum, max, etc.) and sending the result to a designated processing element.
   - **Applications**: Summarizing collective results, such as finding global minima/maxima or calculating the total sum.

6. **All-to-All Communication**:
   - **Description**: Every processing element sends distinct data to every other processing element.
   - **Applications**: Global computations where every processing element needs information from every other, such as certain graph algorithms or matrix computations.

7. **Stencil Communication Pattern** (as previously described):
   - **Description**: Local, neighbor-to-neighbor communication based on a fixed pattern.
   - **Applications**: Grid-based simulations, image processing, numerical solutions to PDEs.

8. **Pipeline Communication Pattern**:
   - **Description**: Processing elements are arranged in a linear sequence, and data is passed from one stage to the next.
   - **Applications**: Sequential processing tasks, such as in signal processing or assembly line simulations.

9. **Ring Communication Pattern**:
   - **Description**: Similar to the pipeline but connects the ends to form a ring.
   - **Applications**: Algorithms that require circular data movement, such as certain sorting algorithms.

10. **Mesh and Hypercube Communication Patterns**:
    - **Description**: Organizing processing elements in multi-dimensional grids or hypercubes, enabling structured communication.
    - **Applications**: Multi-dimensional scientific simulations, parallel algorithms that benefit from spatial locality.

11. **Map Communication Pattern**:
   - **Description**: The map pattern applies the same function or operation to each element in a data set, typically in parallel. Each processing element is responsible for performing the operation on a subset of the data.
   - **Applications**: This pattern is widely used in data parallelism, where the same operation needs to be performed on each data element independently. Examples include applying a filter to an image, parallel array addition, or any operation that can be performed independently on individual data elements.

12. **Transpose Communication Pattern**:
   - **Description**: In the transpose pattern, data is rearranged according to a specific rule, often involving a reordering of dimensions in a multi-dimensional array. This pattern often corresponds to a matrix transpose, where rows become columns and vice versa, but can also be more generally applied to other data transformations.
   - **Applications**: The transpose pattern is common in scientific computing and numerical algorithms that require a change in data layout for efficiency. This includes certain matrix multiplication algorithms, Fast Fourier Transforms (FFTs), and rearranging data to optimize cache usage.

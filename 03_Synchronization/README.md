# Chapter 03: Threads, Synchronization, and Memory - Learning Reflections

**Author**: Tony Fu  
**Date**: August 14, 2023  
**Hardware and Software Configurations**: See [README.md](../README.md) at the repo root

**Reference**: Chapter 3 of [*Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA*](https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA.git) by Bhaumik Vaidya.

## Core Concepts

### 1. A Trick with Large N

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

### 2. Memory Types

There is an informative diagram of memory and a detailed table about the speed, scope, and lifetime of each memory type in Chapter 3 of the book. Below, I will include my version of the table, which contains a bit more detail:

| Memory Type        | Applications             | Speed            | Cached? | Scope                  | Lifetime           | Access Patterns       | Usage                          |
|--------------------|--------------------------|------------------|---------|------------------------|--------------------|-----------------------|--------------------------------|
| **Global Memory**  | General-purpose storage  | Slowest          | Yes     | Host & Device          | Application        | Random/R/W            | General storage; allocation with `cudaMalloc` |
| **Shared Memory**  | Collaborative data within a block | Faster than Global | No | Block                  | Kernel             | Collaborative R/W within block | Declare with `__shared__` in kernel code |
| **Local Memory**   | Thread-specific variables | Same as Global   | Yes     | Thread                 | Kernel             | Private to thread     | Automatic for local variables in kernel |
| **Constant Memory**| Constant variables       | Faster than Global (if cached) | Yes    | Host & Device          | Application        | Read-only             | Declare with `__constant__` and set with `cudaMemcpyToSymbol` |
| **Texture Memory** | Spatially localized data | Faster than Global (if cached) | Yes    | Host & Device          | Application        | Read-only with spatial locality | Bind to texture objects; read with texture fetch functions |
| **Register Memory**| Fastest local storage    | Fastest          | N/A     | Thread                 | Kernel             | Private to thread     | Automatic for local variables in kernel |
| **Pinned Memory**  | Fast host-device transfer | Depends on host  | N/A     | Host                   | Controlled by Host | Host-device transfers | Allocate with `cudaHostAlloc`; enable faster host-device communication |

**Notes**:

- **Cached**: If the memory is cached, it means that a copy of the data is stored in a cache closer to the processing unit, allowing faster access. This is applicable to Constant and Texture Memory, making them faster when there is temporal or spatial locality in access.

- **L1 and L2 Caches**: They are not included in the table because they're not something the programmer typically interacts with directly. We don't allocate space in the L1 or L2 caches or write code to explicitly manage them (unlike, for example, shared or global memory). Instead, they are managed by the hardware and the CUDA runtime, which takes care of caching data as needed based on your memory access patterns.
- **"On-Chip" vs. "Off-Chip"**: The term "on-chip" refers to memory that is physically located on the same semiconductor chip as the GPU's processors. In contrast, off-chip memory like global and local is located in device RAM, which is not on the same chip as the CUDA cores.

Running the code `002_memory_types.cu` gives the following results:
```
Time for global memory: 102 microseconds
Time for shared memory: 54 microseconds
Time for local memory: 53 microseconds
Time for constant memory: 25 microseconds
```

The speed-up in local memory is interesting because local memory is often thought of as a region of global memory that is private to each thread. Local variables that are not statically known to reside in the register file may be placed in local memory (i.e., **register spilling**). However, in this case, the "local" variables are compile-time constants for each thread, so the compiler is likely optimizing them into registers, providing the fastest access time. Registers are the fastest type of memory but have limited capacity.


### 3. Shared Memory and Running Average

The book has erroneously referred to the running average as the moving average. Moving averages operate on a window of fixed size, whereas the running average (a.k.a. cumulative average) considers the sum of all data points up to the current point. `003_running_avg.cu` is my version of the running average. In this section, we will compare the parallelized GPU version with the traditional CPU implementation of the running average.

Here is the GPU version:
```cpp
// Kernel to compute the running average on GPU
__global__ void runningAverageGPU(const float* d_data, float* d_result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sharedSum[1024];

    sharedSum[threadIdx.x] = (tid < N) ? d_data[tid] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = 512; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Store the result
    if (threadIdx.x == 0) {
        atomicAdd(d_result, sharedSum[0]);
    }
}
```
Let's go through this CUDA kernel step by step:

1. **Shared Memory Initialization**:
   ```cpp
   __shared__ float sharedSum[1024];
   sharedSum[threadIdx.x] = (tid < N) ? d_data[tid] : 0;
   __syncthreads();
   ```
   A shared memory array `sharedSum` of size `1024` is declared. Each thread loads a single element from the input array `d_data` into shared memory, based on its thread ID. If the thread ID is outside the bounds of the data array (i.e., `tid >= N`), then it stores `0`. The `__syncthreads()` function is called to synchronize all threads in a block, ensuring that all data has been loaded before proceeding.

2. **Reduction in Shared Memory**:
   ```cpp
   for (int s = 512; s > 0; s >>= 1) {
       if (threadIdx.x < s) {
           sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
       }
       __syncthreads();
   }
   ```
   This loop performs a parallel reduction, which is a common method to sum an array in parallel. The process iteratively adds pairs of elements from the shared memory array, halving the number of active threads in each step. By using a bitwise right shift (`s >>= 1`), the stride `s` is halved in each iteration, leading to a binary tree-like reduction. After each step, the threads are synchronized with `__syncthreads()` to make sure all additions are complete before the next step.

3. **Store the Result**:
   ```cpp
   if (threadIdx.x == 0) {
       atomicAdd(d_result, sharedSum[0]);
   }
   ```
    Since the final result of the reduction is stored in the first element (`sharedSum[0]`), only the thread with `threadIdx.x == 0` needs to perform the atomic addition to the global result variable `d_result`. In the calling code, you divided this result by `N` to obtain the average.

In comparison, here is the CPU version:
```cpp
void runningAverageCPU(const float* h_data, float* result) {
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += h_data[i];
    }
    *result = sum / N;
}
```

The results of the run are:
```
Time for CPU: 25861 microseconds
Time for GPU: 1987 microseconds
CPU Result: 4.87149e+06
GPU Result: 4.99991e+06
```
As expected, the CPU takes much longer to complete the same task than the GPU. However, we also notice that there is a significant error in the CPU result (the correct value should be \((0 + 9,999,999) / 2 = 4,999,999.5\)). This discrepancy could be due to the finite precision of floating-point numbers, which may accumulate rounding errors throughout the computation.


### 4. Atomic Operations
Atomic operations are operations that complete in a single uninterrupted step relative to other threads. In CUDA, atomic operations are supported for both global and shared memory:

1. **`atomicAdd(int* address, int value)` or `atomicAdd(float* address, float value)`**: Atomically adds a value to an integer or floating-point number in memory.
2. **`atomicSub(int* address, int value)`**: Atomically subtracts a value from an integer in memory.
3. **`atomicExch(T* address, T value)`**: Atomically exchanges two values in memory, where `T` can be integer or float.
4. **`atomicMin(int* address, int value)` or `atomicMin(unsigned int* address, unsigned int value)`**: Atomically computes the minimum of two integers or unsigned integers in memory.
5. **`atomicMax(int* address, int value)` or `atomicMax(unsigned int* address, unsigned int value)`**: Atomically computes the maximum of two integers or unsigned integers in memory.
6. **`atomicInc(unsigned int* address, unsigned int val)`**: Atomically increments an unsigned integer by one, wrapping to zero if the incremented result is greater than a specified value.
7. **`atomicDec(unsigned int* address, unsigned int val)`**: Atomically decrements an unsigned integer by one, saturating at zero.
8. **`atomicCAS(int* address, int compare, int val)`**: (Compare And Swap) Compares a value in memory with a specified value and, only if they are the same, modifies the memory value.

### 5. Texture Memory

CUDA texture objects replaced CUDA texture references starting with the CUDA SDK 5.0 toolkit in 2012, paving the way for a more object-oriented approach to using texture memory. This led to the deprecation of texture references in April 2021 (CUDA SDK 11.3) and their removal in CUDA SDK 12. Below is a modernized version of the example written using CUDA texture objects:

1. **Include Statements and Macro Definitions**:
   ```cpp
   #include <cuda_runtime.h>

   #define NUM_THREADS 10
   #define N 10
   ```

2. **Kernel Definition**:
   ```cpp
   __global__ void gpu_texture_memory(int n, cudaTextureObject_t textureObj, float* d_out)
   {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n)
       {
           float temp = tex1Dfetch<float>(textureObj, idx);
           d_out[idx] = temp;
       }
   }
   ```
   Here, the texture object `textureObj` is read from, and the result is written to the output array `d_out`. Unlike global memory, textures require specialized functions to access them. Since our texture is 1D, we use the 1D accessor. There are also 2D and 3D options.

3. **Main Function - Initialization**:
   ```cpp
   float* d_out;
   cudaMalloc((void**)&d_out, sizeof(float) * N);

   float* h_out = (float*)malloc(sizeof(float) * N);
   float h_in[N];
   for (int i = 0; i < N; i++)
   {
       h_in[i] = float(i);
   }
   ```
   Memory is allocated for the device output array `d_out`, the host output array `h_out`, and the input array `h_in` is initialized.

4. **Device Memory Allocation and Copy for Input Array**:
   ```cpp
   float* d_in;
   cudaMalloc((void**)&d_in, sizeof(float) * N);
   cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice);
   ```
   Memory is allocated for the device input array `d_in`, and the input data from the host is copied to it. Note that the `d_in` array data is first placed in global memory. A direct copy from host to texture memory is not supported. We must first copy the host array to global memory, then enable access to that memory through the texture cache.

5. **Texture Object Creation**:
   Creating a texture object is a complex process. It involves relating global memory data to the texture cache and requires the creation of two configuration objects: `cudaResourceDesc` and `cudaTextureDesc`. The former describes the properties of the data:
   ```cpp
   cudaResourceDesc resDesc;
   memset(&resDesc, 0, sizeof(resDesc));
   resDesc.resType = cudaResourceTypeLinear;
   resDesc.res.linear.devPtr = d_in;
   resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
   resDesc.res.linear.desc.x = 32; // bits per channel
   resDesc.res.linear.sizeInBytes = sizeof(float) * N;
   ```
   The latter controls how the data is accessed through the texture cache, such as the reading mode:
   ```cpp
   cudaTextureDesc texDesc;
   memset(&texDesc, 0, sizeof(texDesc));
   texDesc.readMode = cudaReadModeElementType;
   ```
   In addition to the two configuration objects, a handle for the texture object must be created. It is customary to initialize it to 0 before use:
   ```cpp
   cudaTextureObject_t textureObj = 0;
   cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, NULL);
   ```

6. **Kernel Execution**:
   ```cpp
   int num_blocks = N / NUM_THREADS + ((N % NUM_THREADS) ? 1 : 0);
   gpu_texture_memory << <num_blocks, NUM_THREADS >> > (N, textureObj, d_out);
   ```
   The GPU kernel is launched with the specified number of blocks and threads, and the texture object is passed as an argument.

7. **Copying Results to Host and Printing**:
   ```cpp
   cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

   printf("Use of Texture memory on GPU: \n");
   for (int i = 0; i < N; i++) {
       printf("Texture element at %d is : %f\n", i, h_out[i]);
   }
   ```
   The results are copied back to the host and printed to the console.

8. **Cleanup**:
   ```cpp
   free(h_out);
   cudaFree(d_in);
   cudaFree(d_out);
   cudaDestroyTextureObject(textureObj);
   ```
   Don't forget to free all allocated memory and destroy the texture object.

### 6. Dot Product

Take a look at [`009_dot_product.cu`](009_dot_product.cu), where you'll find this kernel function:

```cpp
__global__ void gpu_dot(float* d_a, float* d_b, float* d_c)
{
	__shared__ float partial_sum[THREADS_PER_BLOCK];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int index = threadIdx.x;

	// Compute element-wise dot product
	float sum = 0;
	while (tid < N)
	{
		sum += d_a[tid] * d_b[tid];
		tid += blockDim.x * gridDim.x;
	}

	partial_sum[index] = sum;

	__syncthreads();

	// Reduction
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (index < i)
		{
			partial_sum[index] += partial_sum[index + i];
		}
		__syncthreads();
		i >>= 1;
	}

	if (index == 0)
	{
		d_c[blockIdx.x] = partial_sum[0];
	}
}
```

Here, the `partial_sum` array is a shared memory within each block, used to store the element-wise multiplication of the `d_a` and `d_b` arrays stored in global memory. This shared memory can enable faster access due to its closeness to the processor and may leverage spatial locality (where adjacent threads access adjacent data), though whether it will be cached depends on the specific GPU architecture.

Next, a reduction operation takes place to combine the products in pairs in a binary-search-like fashion until the whole `partial_sum` array has been reduced to a single sum at index 0. This block-wise partial sum is then assigned to `d_c`.

The author of the code has chosen to move the final summation to the host rather than using atomic addition on the GPU. While atomic addition would provide a way to sum all elements in `d_c` to get the final sum, it could lead to contention and thus might be slower than summing the partial results on the CPU. See the following host code for how the final summation is carried out:

```cpp
int main()
{
	// Initialization code (not shown)

	partial_sum = (float*)malloc(blocks_per_grid * sizeof(float));
	cudaMalloc((void**)&d_partial_sum, blocks_per_grid * sizeof(float));

	gpu_dot << <blocks_per_grid, THREADS_PER_BLOCK >> > (d_a, d_b, d_partial_sum);

	cudaMemcpy(partial_sum, d_partial_sum, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);

	// Calculate final dot product on host
	h_c = 0;
	for (int i = 0; i < blocks_per_grid; i++) {
		h_c += partial_sum[i];
	}
	
	// Rest of the code...
}
```

### 7. Matrix Multiplication

Let's consider the multiplication of two 4 x 4 matrices. If we just use global memory, each number in either matrix would need to be accessed exactly 4 times in the process, leading to a significant amount of traffic to global memory. This presents an opportunity to optimize performance by taking advantage of shared memory.

Here's an example code in [`010_gpu_matmul.cu`](010_gpu_matmul.cu) that demonstrates this approach:

```cpp
#define TILE_SIZE 2
const int size = 4;

__global__ void gpu_matmul_shared(float* d_a, float* d_b, float* d_c, const int size)
{
	__shared__ float shared_a[TILE_SIZE][TILE_SIZE];
	__shared__ float shared_b[TILE_SIZE][TILE_SIZE];

	int col = TILE_SIZE * blockIdx.x + threadIdx.x;
	int row = TILE_SIZE * blockIdx.y + threadIdx.y;

	for (int i = 0; i < size / TILE_SIZE; i++)
	{
		shared_a[threadIdx.y][threadIdx.x] = d_a[row * size + (i * TILE_SIZE + threadIdx.x)];
		shared_b[threadIdx.y][threadIdx.x] = d_b[(i * TILE_SIZE + threadIdx.y) * size + col];

		__syncthreads();

		for (int j = 0; j < TILE_SIZE; j++)
		{
			d_c[row * size + col] += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
		}
		__syncthreads();
	}
}
```

In this code, we're dividing the resulting 4 x 4 matrix into four 2 x 2 (TILE_SIZE x TILE_SIZE) sub-matrices, with each residing in a separate block. The `col` and `row` indices are calculated to figure out what column and row the current thread corresponds to in the final 4 x 4 matrix.

Unlike other arrays in this code, `d_a` and `d_b` are 1D arrays that represent 2D data in row-major order. Each block copies a 2 x 2 submatrix to `shared_a` and `shared_b`. The inner for-loop then takes advantage of shared memory, as each thread performs a mini matrix multiplication between the two sub-matrices in `shared_a` and `shared_b`, without having to query the matrix values multiple times from global memory. The process then repeats for other pairs of sub-matrices.
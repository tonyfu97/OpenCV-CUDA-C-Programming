# Chapter 01: Introduction to CUDA - Learning Reflections

**Author**: Tony Fu  
**Date**: August 14, 2023  
**Hardware and Software Configurations**: See README.md at the repo root

**Reference**: Chapter 1 of [*Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA*](https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA.git) by Bhaumik Vaidya.

## Core Concepts

### 1. Set up Visual Studio Project

* Inside the repository root, within VS, go to File -> New -> Project... . Select CUDA 12.2 Runtime.
* Name the new project `01_IntroductionCuda`.
* Specify the "Location" to the repo root.
* In "Solution," select "Create new solution." This is used to group one or more related projects.
* Check "Place solution and project in the same directory."
* A new directory will be created, and a new editor window will open. Delete the auto-generated `kernel.cu` and add `Hello_Cuda.cu`.
* Open `Hello_CUDA.cu` and click the run arrow at the top. The code should be built, run, and print "Hello CUDA!"
* Remember to add additional extensions in `.gitignore` to avoid bloated version control.

### 2. Host vs. Device

- **Host**: The CPU, where regular C/C++ code runs.
- **Device**: The GPU, where CUDA code (typically written in `.cu` extension) runs. Compiled by the NVIDIA CUDA compiler (`nvcc`).

### 3. CUDA Code Example

Below is the simple CUDA code example from the book:

```cpp
#include <stdio.h>
__global__ void myfirstkernel(void) {
}

int main(void) {
	myfirstkernel << <1, 1 >> > ();
	printf("Hello, CUDA!\n");
	return 0;
}
```
* `__global__ void myfirstkernel(void) { }`: This is a definition of a kernel function that can be executed on the GPU. The `__global__` specifier indicates that this function can be called from host code but runs on the device. It's an empty kernel, so it doesn't perform any computation. This part is considered device code.
* `int main(void) { ... }`: This is the main function of the program, and it's executed on the host (CPU). Within this function:
  * `myfirstkernel <<<1, 1>>> ();`: This line launches the kernel on the GPU. The syntax `<<<1, 1>>>` specifies the launch configuration, with one thread block and one thread per block.
  * `printf("Hello, CUDA!\n");`: This line prints a message from the host. The entire main function is considered host code.

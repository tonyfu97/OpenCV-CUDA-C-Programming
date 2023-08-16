#include <iostream>
#include <cuda_runtime.h>

#define NUM_THREADS 10
#define N 10

__global__ void gpu_texture_memory(int n, cudaTextureObject_t textureObj, float* d_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float temp = tex1Dfetch<float>(textureObj, idx);
        d_out[idx] = temp;
    }
}

int main()
{
    float* d_out;
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    float* h_out = (float*)malloc(sizeof(float) * N);
    float h_in[N];

    for (int i = 0; i < N; i++)
    {
        h_in[i] = float(i);
    }

    // Allocate device memory for the input array and copy data to it
    float* d_in;
    cudaMalloc((void**)&d_in, sizeof(float) * N);
    cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_in;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = sizeof(float) * N;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t textureObj = 0;
    cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, NULL);

    int num_blocks = N / NUM_THREADS + ((N % NUM_THREADS) ? 1 : 0);
    gpu_texture_memory << <num_blocks, NUM_THREADS >> > (N, textureObj, d_out);

    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    printf("Use of Texture memory on GPU: \n");
    for (int i = 0; i < N; i++) {
        printf("Texture element at %d is : %f\n", i, h_out[i]);
    }

    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaDestroyTextureObject(textureObj);
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

cudaError_t mulMatricesGPU(int *c, const int *a, const int *b, unsigned int size, int N, clock_t *timer);

__global__ void addKernel(int *c, const int *a, const int *b, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

void mulMatricesCPU(int* c, const int* a, const int* b, int N) {
    for (int row = 0; row < N; row++) 
        for (int col = 0; col < N; col++) {
            int sum = 0;
            for (int k = 0; k < N; k++) 
                sum += a[row * N + k] * b[k * N + col];
            c[row * N + col] = sum;
        }
}

int main()
{
    int N;
    printf("Enter the N*N matrix size: ");
    scanf("%d", &N);
    int size = N * N * sizeof(float);
    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c_gpu = (int*)malloc(size);
    int* c_cpu = (int*)malloc(size);

    srand((unsigned)time(NULL));
    for (int i = 0; i < N * N; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    clock_t timer;

    cudaError_t cudaStatus = mulMatricesGPU(c_gpu, a, b, size, N, &timer);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mulMatricesGPU failed!");
        return 1;
    }
    double gpu_time = (double)(clock() - timer) / CLOCKS_PER_SEC;

    timer = clock();
    mulMatricesCPU(c_cpu, a, b, N);
    double cpu_time = (double)(clock() - timer) / CLOCKS_PER_SEC;

    int match = 1;
    for (int i = 0; i < N * N; i++) {
        if (c_gpu[i] != c_cpu[i]) {
            match = 0;
            break;
        }
    }

    printf("GPU: %.4f s\n", gpu_time);
    printf("CPU: %.4f s\n", cpu_time);
    printf("The results %s\n", match ? "match!" : " DON'T match!");

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t mulMatricesGPU(int *c, const int *a, const int *b, unsigned int size, int N, clock_t *timer)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threads(16, 16);
    dim3 blockDim((N + 15) / 16, (N + 15) / 16);

    *timer = clock();
    addKernel<<<blockDim, threads>>>(dev_c, dev_a, dev_b, N);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

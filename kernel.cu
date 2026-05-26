#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define WIDTH 2024
#define HEIGHT 2024
#define BLOCK_SIZE 16

cudaError_t blurCuda(unsigned char* input, unsigned char* output, int width, int height, int size, clock_t* timer);

__global__ void blurGPU(unsigned char* input, unsigned char* output, int w, int h)
{
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (row < h && col < w)
        tile[ty][tx] = input[row * w + col];
    else
        tile[ty][tx] = 0;

    if (threadIdx.x == 0) {
        if (col > 0 && row < h)
            tile[ty][0] = input[row * w + (col - 1)];
        else
            tile[ty][0] = 0;
    }

    if (threadIdx.x == blockDim.x - 1 || col == w - 1)
    {
        if (col + 1 < w && row < h)
            tile[ty][tx + 1] = input[row * w + (col + 1)];

        else
            tile[ty][tx + 1] = 0;
    }

    if (threadIdx.y == 0) {
        if (row > 0 && col < w)
            tile[0][tx] = input[(row - 1) * w + col];
        else
            tile[0][tx] = 0;
    }

    if (threadIdx.y == blockDim.y - 1 || row == h - 1) {
        if (row + 1 < h && col < w)
            tile[ty + 1][tx] = input[(row + 1) * w + col];
        else
            tile[ty + 1][tx] = 0;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[0][0] = (row > 0 && col > 0) ? input[(row - 1) * w + (col - 1)] : 0;
    }

    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        tile[0][tx + 1] = (row > 0 && col + 1 < w) ? input[(row - 1) * w + (col + 1)] : 0;
    }

    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        tile[ty + 1][0] = (row + 1 < h && col > 0) ? input[(row + 1) * w + (col - 1)] : 0;
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        tile[ty + 1][tx + 1] = (row + 1 < h && col + 1 < w) ? input[(row + 1) * w + (col + 1)] : 0;
    }

    __syncthreads();

    if (row < h && col < w)
    {
        int sum = 0;
        int count = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = row + dy;
                int nx = col + dx;

                if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                    sum += tile[ty + dy][tx + dx];
                    count++;
                }
            }
        }
        output[row * w + col] = (unsigned char)(sum / count);
    }
}

void blurFilterCPU(unsigned char* input, unsigned char* output, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int sum = 0;
            int count = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = row + dy;
                    int nx = col + dx;

                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        sum += input[ny * width + nx];
                        count++;
                    }
                }
            }

            output[row * width + col] = (unsigned char)(sum / count);
        }
    }
}

int main()
{
    int size = WIDTH * HEIGHT * sizeof(unsigned char);

    unsigned char* h_in = (unsigned char*)malloc(size);
    unsigned char* h_out_cpu = (unsigned char*)malloc(size);
    unsigned char* h_out_gpu = (unsigned char*)malloc(size);

    srand((unsigned)time(NULL));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_in[i] = (unsigned char)(rand() % 256);
    }

    clock_t start = clock();
    blurFilterCPU(h_in, h_out_cpu, WIDTH, HEIGHT);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC;

    cudaError_t cudaStatus = blurCuda(h_in, h_out_gpu, WIDTH, HEIGHT, size, &start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    double
        gpu_time = (double)(clock() - start) / CLOCKS_PER_SEC;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    int match = 1;
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        if (h_out_cpu[i] != h_out_gpu[i]) {
            match = 0;
            break;
        }
    }

    printf("Image resolution: %dx%d\n", WIDTH, HEIGHT);
    printf("CPU: %.4f s\n", cpu_time);
    printf("GPU: %.4f s\n", gpu_time);
    printf("The results %s\n", match ? "match!" : "DON'T match!");

    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);

    return 0;
}

cudaError_t blurCuda(unsigned char* input, unsigned char* output, int width, int height, int size, clock_t* timer)
{
    unsigned char* d_in, * d_out;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_in, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_out, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    *timer = clock();
    blurGPU <<<blocks, threads>>> (d_in, d_out, WIDTH, HEIGHT);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "blurGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_in);
    cudaFree(d_out);

    return cudaStatus;
}

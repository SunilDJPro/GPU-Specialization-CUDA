#ifndef MERGE_SORT_H
#define MERGE_SORT_H

#include <iostream>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <tuple>

// CUDA error checking macro (replaces helper_cuda.h functionality)
#define checkCudaErrors(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function declarations
__host__ long* mergesort(long *data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid);
__host__ long *generateRandomLongArray(int numElements);
__host__ std::tuple<long *, long *, dim3 *, dim3 *> allocateMemory(int numElements);
__host__ void printHostMemory(long *host_mem, int num_elments);
__host__ std::tuple<dim3, dim3, int> parseCommandLineArguments(int argc, char **argv);

__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks);
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end);

#endif // MERGE_SORT_H
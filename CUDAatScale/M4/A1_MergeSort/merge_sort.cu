#include "merge_sort.h"
#include <string>

#define min(a, b) (a < b ? a : b)
// Based on https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu

__host__ std::tuple<dim3, dim3, int> parseCommandLineArguments(int argc, char** argv) 
{
    int numElements = 32;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'n':
                    i++;
                    numElements = atoi(argv[i]);
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }
    return {threadsPerBlock, blocksPerGrid, numElements};
}

__host__ long *generateRandomLongArray(int numElements)
{
    // Generate random array of long integers of size numElements
    long *randomLongs = (long*)malloc(numElements * sizeof(long));
    
    // Seed random number generator
    srand(time(NULL));
    
    // Generate random values
    for(int i = 0; i < numElements; i++) {
        randomLongs[i] = rand() % 1000; // Random values 0-999
    }

    return randomLongs;
}

__host__ void printHostMemory(long *host_mem, int num_elments)
{
    // Output results
    for(int i = 0; i < num_elments; i++)
    {
        printf("%ld ",host_mem[i]);
    }
    printf("\n");
}

__host__ int main(int argc, char** argv) 
{

    auto[threadsPerBlock, blocksPerGrid, numElements] = parseCommandLineArguments(argc, argv);

    long *data = generateRandomLongArray(numElements);

    printf("Unsorted data: ");
    printHostMemory(data, numElements);

    data = mergesort(data, numElements, threadsPerBlock, blocksPerGrid);

    printf("Sorted data: ");
    printHostMemory(data, numElements);
    
    free(data);
    return 0;
}

__host__ std::tuple <long* ,long* ,dim3* ,dim3*> allocateMemory(int numElements)
{
    long *D_data, *D_swp;
    dim3 *D_threads, *D_blocks;
    
    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    
    // Actually allocate the two arrays
    checkCudaErrors(cudaMalloc((void**)&D_data, numElements * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**)&D_swp, numElements * sizeof(long)));

    // Copy the thread / block info to the GPU as well
    checkCudaErrors(cudaMalloc((void**)&D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**)&D_blocks, sizeof(dim3)));

    return {D_data, D_swp, D_threads, D_blocks};
}

__host__ long* mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    auto[D_data, D_swp, D_threads, D_blocks] = allocateMemory(size);

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    // Copy input data to GPU
    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice));
    
    // Copy thread and block dimensions to GPU
    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    // Initialize timing metrics variables
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    checkCudaErrors(cudaEventRecord(start));

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);
        
        checkCudaErrors(cudaDeviceSynchronize());

        // Switch the input / output arrays instead of copying them around
        long* temp = A;
        A = B;
        B = temp;
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    
    // Calculate and print kernel execution time
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost));

    // Free the GPU memory
    checkCudaErrors(cudaFree(D_data));
    checkCudaErrors(cudaFree(D_swp));
    checkCudaErrors(cudaFree(D_threads));
    checkCudaErrors(cudaFree(D_blocks));
    
    // Clean up events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return data;
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    
    // Initialize 3 long variables start, middle, and end
    long start, middle, end;
    // start is set to the width of the merge sort data span * the thread index * number of slices
    start = width * idx * slices;

    for (long slice = 0; slice < slices; slice++) {
        // Break from loop when the start variable is >= size of the input array
        if (start >= size) break;

        // Set middle to be minimum middle index (start index plus 1/2 width) and the size of the input array
        middle = min(start + (width >> 1), size);

        // Set end to the minimum of the end index (start index plus the width of the current data window) and the size of the input array
        end = min(start + width, size);
       
        // Perform bottom up merge given the two available arrays and the start, middle, and end variables
        gpu_bottomUpMerge(source, dest, start, middle, end);
        
        // Increase the start index by the width of the current data window
        start += width;
    }
}

//
// Finally, sort something gets called by gpu_mergesort() for each slice
// Note that the pseudocode below is not necessarily 100% complete you may want to review the merge sort algorithm.
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;

    // Create a for loop that iterates between the start and end indexes
    for (long k = start; k < end; k++) {
        // if i is before the middle index and (j is the final index or the value at i <  the value at j)
        if (i < middle && (j >= end || source[i] <= source[j])) {
            // set the value in the destination array at index k to the value at index i in the source array
            dest[k] = source[i];
            // increment i
            i++;
        } else {
            // set the value in the destination array at index k to the value at index j in the source array
            dest[k] = source[j];
            // increment j
            j++;
        }
    }
}
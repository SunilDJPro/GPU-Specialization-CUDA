/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include "simpleLinearBlurFilter.hpp"

// Device memory pointers
__device__ uchar *d_r, *d_g, *d_b;
__device__ uchar *d_r_out, *d_g_out, *d_b_out;

/*
 * CUDA Kernel Device code
 *
 */
__global__ void applySimpleLinearBlurFilter(uchar *r, uchar *g, uchar *b)
{
    // Calculate thread ID
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int num_image_pixels = d_rows * d_columns;
    
    // Shared memory for storing pixel values
    extern __shared__ uchar shared_mem[];
    uchar *shared_r = shared_mem;
    uchar *shared_g = &shared_mem[blockDim.x + 2];
    uchar *shared_b = &shared_mem[2 * (blockDim.x + 2)];
    
    if(threadId < num_image_pixels)
    {
        // Calculate row and column from thread ID
        int row = threadId / d_columns;
        int col = threadId % d_columns;
        
        // Load data into shared memory with padding for left and right neighbors
        int shared_idx = threadIdx.x + 1; // +1 for left padding
        
        // Load current pixel
        shared_r[shared_idx] = r[threadId];
        shared_g[shared_idx] = g[threadId];
        shared_b[shared_idx] = b[threadId];
        
        // Load left neighbor (handle edge case)
        if (threadIdx.x == 0) {
            if (col > 0) {
                shared_r[0] = r[threadId - 1];
                shared_g[0] = g[threadId - 1];
                shared_b[0] = b[threadId - 1];
            } else {
                // Left edge - use current pixel value
                shared_r[0] = r[threadId];
                shared_g[0] = g[threadId];
                shared_b[0] = b[threadId];
            }
        }
        
        // Load right neighbor (handle edge case)
        if (threadIdx.x == blockDim.x - 1) {
            if (col < d_columns - 1) {
                shared_r[shared_idx + 1] = r[threadId + 1];
                shared_g[shared_idx + 1] = g[threadId + 1];
                shared_b[shared_idx + 1] = b[threadId + 1];
            } else {
                // Right edge - use current pixel value
                shared_r[shared_idx + 1] = r[threadId];
                shared_g[shared_idx + 1] = g[threadId];
                shared_b[shared_idx + 1] = b[threadId];
            }
        }
        
        // Sync threads so that shared memory is fully loaded
        __syncthreads();
        
        // Apply simple 3-pixel wide linear blur filter
        // Average the current pixel with its left and right neighbors
        uchar blurred_r, blurred_g, blurred_b;
        
        if (col == 0) {
            // Left edge: average current and right pixel
            blurred_r = (shared_r[shared_idx] + shared_r[shared_idx + 1]) / 2;
            blurred_g = (shared_g[shared_idx] + shared_g[shared_idx + 1]) / 2;
            blurred_b = (shared_b[shared_idx] + shared_b[shared_idx + 1]) / 2;
        } else if (col == d_columns - 1) {
            // Right edge: average left and current pixel
            blurred_r = (shared_r[shared_idx - 1] + shared_r[shared_idx]) / 2;
            blurred_g = (shared_g[shared_idx - 1] + shared_g[shared_idx]) / 2;
            blurred_b = (shared_b[shared_idx - 1] + shared_b[shared_idx]) / 2;
        } else {
            // Middle pixels: average left, current, and right pixels
            blurred_r = (shared_r[shared_idx - 1] + shared_r[shared_idx] + shared_r[shared_idx + 1]) / 3;
            blurred_g = (shared_g[shared_idx - 1] + shared_g[shared_idx] + shared_g[shared_idx + 1]) / 3;
            blurred_b = (shared_b[shared_idx - 1] + shared_b[shared_idx] + shared_b[shared_idx + 1]) / 3;
        }
        
        // Sync threads before writing results
        __syncthreads();
        
        // Write blurred values back to global memory
        r[threadId] = blurred_r;
        g[threadId] = blurred_g;
        b[threadId] = blurred_b;
    }
}

__host__ float compareColorImages(uchar *r0, uchar *g0, uchar *b0, uchar *r1, uchar *g1, uchar *b1, int rows, int columns)
{
    cout << "Comparing actual and test pixel arrays\n";
    int numImagePixels = rows * columns;
    int imagePixelDifference = 0.0;

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            // Fixed indexing bug: should be r*columns+c, not r*rows+c
            uchar image0R = r0[r*columns+c];
            uchar image0G = g0[r*columns+c];
            uchar image0B = b0[r*columns+c];
            uchar image1R = r1[r*columns+c];
            uchar image1G = g1[r*columns+c];
            uchar image1B = b1[r*columns+c];
            imagePixelDifference += ((abs(image0R - image1R) + abs(image0G - image1G) + abs(image0B - image1B))/3);
        }
    }

    float meanImagePixelDifference = imagePixelDifference / numImagePixels;
    float scaledMeanDifferencePercentage = (meanImagePixelDifference / 255);
    printf("meanImagePixelDifference: %f scaledMeanDifferencePercentage: %f\n", meanImagePixelDifference, scaledMeanDifferencePercentage);
    return scaledMeanDifferencePercentage;
}

__host__ void allocateDeviceMemory(int rows, int columns)
{
    //Allocate device constant symbols for rows and columns
    cudaMemcpyToSymbol(d_rows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_columns, &columns, sizeof(int), 0, cudaMemcpyHostToDevice);
}

__host__ void executeKernel(uchar *r, uchar *g, uchar *b, int rows, int columns, int threadsPerBlock)
{
    cout << "Executing kernel\n";
    //Launch the convert CUDA Kernel
    int blocksPerGrid = (rows * columns + threadsPerBlock - 1) / threadsPerBlock; // Ceiling division
    
    // Calculate shared memory size needed: 3 arrays * (threadsPerBlock + 2 for padding)
    size_t sharedMemSize = 3 * (threadsPerBlock + 2) * sizeof(uchar);

    applySimpleLinearBlurFilter<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(r, g, b);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    cout << "Cleaning CUDA device\n";
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
    cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 256;
    std::string inputImage = "sloth.png";
    std::string outputImage = "grey-sloth.png";
    std::string currentPartId = "test";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option.compare("-p") == 0)
        {
            currentPartId = value;
        }
    }
    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " currentPartId: " << currentPartId << " threadsPerBlock: " << threadsPerBlock << "\n";
    return {inputImage, outputImage, currentPartId, threadsPerBlock};
}

__host__ std::tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile)
{
    cout << "Reading Image From File\n";
    Mat img = imread(inputFile, IMREAD_COLOR);
    
    const int rows = img.rows;
    const int columns = img.cols;
    size_t size = sizeof(uchar) * rows * columns;

    cout << "Rows: " << rows << " Columns: " << columns << "\n";

    uchar *r, *g, *b;
    cudaMallocManaged(&r, size);
    cudaMallocManaged(&g, size);
    cudaMallocManaged(&b, size);
    
    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < columns; ++x)
        {
            Vec3b rgb = img.at<Vec3b>(y, x);
            // Fixed indexing: should be y*columns+x, not y*rows+x
            r[y*columns+x] = rgb.val[2]; // Red channel
            g[y*columns+x] = rgb.val[1]; // Green channel  
            b[y*columns+x] = rgb.val[0]; // Blue channel
        }
    }

    return {rows, columns, r, g, b};
}

__host__ std::tuple<uchar *, uchar *, uchar *>applyBlurKernel(std::string inputImage)
{
    cout << "CPU applying kernel\n";
    Mat img = imread(inputImage, IMREAD_COLOR);
    const int rows = img.rows;
    const int columns = img.cols;

    uchar *r = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *g = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *b = (uchar *)malloc(sizeof(uchar) * rows * columns);

    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < columns; ++x)
        {
            if (x == 0) {
                // Left edge: average current and right pixels
                Vec3b rgb1 = img.at<Vec3b>(y, x);
                Vec3b rgb2 = img.at<Vec3b>(y, x+1);
                b[y*columns+x] = (rgb1[0] + rgb2[0])/2;
                g[y*columns+x] = (rgb1[1] + rgb2[1])/2;
                r[y*columns+x] = (rgb1[2] + rgb2[2])/2;
            } else if (x == columns-1) {
                // Right edge: average left and current pixels
                Vec3b rgb0 = img.at<Vec3b>(y, x-1);
                Vec3b rgb1 = img.at<Vec3b>(y, x);
                b[y*columns+x] = (rgb0[0] + rgb1[0])/2;
                g[y*columns+x] = (rgb0[1] + rgb1[1])/2;
                r[y*columns+x] = (rgb0[2] + rgb1[2])/2;
            } else {
                // Middle pixels: average left, current, and right pixels
                Vec3b rgb0 = img.at<Vec3b>(y, x-1);
                Vec3b rgb1 = img.at<Vec3b>(y, x);
                Vec3b rgb2 = img.at<Vec3b>(y, x+1);
                b[y*columns+x] = (rgb0[0] + rgb1[0] + rgb2[0])/3;
                g[y*columns+x] = (rgb0[1] + rgb1[1] + rgb2[1])/3;
                r[y*columns+x] = (rgb0[2] + rgb1[2] + rgb2[2])/3;
            }
        }
    }

    return {r, g, b};
}

int main(int argc, char *argv[])
{
    std::tuple<std::string, std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    std::string currentPartId = get<2>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<3>(parsedCommandLineArgsTuple);
    try 
    {
        auto[rows, columns, r, g, b] = readImageFromFile(inputImage);

        allocateDeviceMemory(rows, columns);
        executeKernel(r, g, b, rows, columns, threadsPerBlock);

        Mat colorImage(rows, columns, CV_8UC3);
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < columns; ++x)
            {
                // Fixed indexing: should be y*columns+x, not y*rows+x
                colorImage.at<Vec3b>(y,x) = Vec3b(b[y*columns+x], g[y*columns+x], r[y*columns+x]);
            }
        }

        imwrite(outputImage, colorImage, compression_params);

        auto[test_r, test_g, test_b] = applyBlurKernel(inputImage);
        
        float scaledMeanDifferencePercentage = compareColorImages(r, g, b, test_r, test_g, test_b, rows, columns) * 100;
        cout << "Mean difference percentage: " << scaledMeanDifferencePercentage << "\n";

        // Free managed memory
        cudaFree(r);
        cudaFree(g);
        cudaFree(b);
        
        // Free CPU memory
        free(test_r);
        free(test_g);
        free(test_b);

        cleanUpDevice();
    }
    catch (Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}
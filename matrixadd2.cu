#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Size of the matrix (N x N)

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        C[i][j] = A[i][j] + B[i][j];
    }
}

int main() {
    // Host matrices
    float (*h_A)[N], (*h_B)[N], (*h_C)[N];

    // Device matrices
    float (*d_A)[N], (*d_B)[N], (*d_C)[N];

    // Size of the matrices in bytes
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    h_A = (float (*)[N])malloc(size);
    h_B = (float (*)[N])malloc(size);
    h_C = (float (*)[N])malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        printf("Host memory allocation failed\n");
        return -1;
    }

    // Initialize host matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i][j] = (float)(i + j);   // Sample initialization
            h_B[i][j] = (float)(i * j);   // Sample initialization
        }
    }

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        printf("Device memory allocation failed for d_A: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        printf("Device memory allocation failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return -1;
    }
    
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        printf("Device memory allocation failed for d_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    printf("Launching kernel with grid dimensions: %d x %d blocks, each with %d x %d threads\n", 
        numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // Check for errors in kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        return -1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify the result (check a few elements for large matrices)
    bool correct = true;
    int checkLimit = (N < 100) ? N : 100; // Limit checking for very large matrices
    
    for (int i = 0; i < checkLimit; i++) {
        for (int j = 0; j < checkLimit; j++) {
            if (fabs(h_A[i][j] + h_B[i][j] - h_C[i][j]) > 1e-5) {
                printf("Error at position (%d, %d)!\n", i, j);
                printf("Expected: %f, Got: %f\n", h_A[i][j] + h_B[i][j], h_C[i][j]);
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }
    
    if (correct) {
        printf("Matrix addition completed successfully!\n");
    }
    
    // Print a small sample of the result (first 4x4 elements)
    printf("Sample results (first 4x4 elements):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("C[%d][%d] = %.2f = A[%d][%d] = %.2f + B[%d][%d] = %.2f\n", i, j, h_C[i][j], i, j, h_A[i][j], i, j, h_B[i][j]);
        }
        printf("\n");
    }
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
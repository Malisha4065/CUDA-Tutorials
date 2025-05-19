#include <stdio.h>
#include <cuda_runtime.h>

#define N 16    // Size of the matrix (N x N)

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main() {
    // Host matrices
    float h_A[N][N], h_B[N][N], h_C[N][N];

    // Device matrices
    float (*d_A)[N], (*d_B)[N], (*d_C)[N];

    // Size of the matrices in bytes
    size_t size = N * N * sizeof(float);

    // Initialize host matrices with sample data
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i][j] = i + j;  // Sample initialization
            h_B[i][j] = i * j;  // Sample initialization
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    bool correct = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(h_A[i][j] + h_B[i][j] - h_C[i][j]) > 1e-5) {
                printf("Error at position (%d, %d)!\n", i, j);
                printf("Expected: %f, Got: %f\n", h_A[i][j] + h_B[i][j], h_C[i][j]);
                correct = false;
                break;
            }
        }
    }

    if (correct) {
        printf("Matrix addition completed successfully!\n");
    }

    // Print a sample of the result (first 4x4 elements)
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
    
    return 0;
}
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    // Vector size
    int N = 1024;
    size_t size = N * sizeof(float);

    // Host vectors
    float *h_A, *h_B, *h_C;

    // Device vectors
    float *d_A, *d_B, *d_C;

    // Allocate memory for host vectors
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory for device vectors
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy host vectors to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    // Check for error in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err  != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            printf("Verification failed at index %d!\n", i);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful!\n");
    }

    // Print first few results as sample
    printf("Sample results:\n");
    for (int i = 0; i < 5; i++) {
        printf("A[%d] = %.2f, B[%d] = %.2f, C[%d] = %.2f\n",
            i, h_A[i], i, h_B[i], i, h_C[i]);
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
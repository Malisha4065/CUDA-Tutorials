#include <stdio.h>

__global__ void cuda_hello() {
    printf("hello from gpu\n");
}

int main() {
    cuda_hello <<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
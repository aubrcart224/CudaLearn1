
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstdlib>



__global__ void testKernel(float* a) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    a[index] = index;
}

int main() {
    float* d_a;
    cudaMalloc(&d_a, sizeof(float) * 256);
    testKernel << <1, 256 >> > (d_a);
    cudaFree(d_a);
    return 0;

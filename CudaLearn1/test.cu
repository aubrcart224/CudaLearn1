#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstdlib>



// Kernel function to add the elements of two arrays
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) {
		C[i] = A[i] + B[i];
	}

}

// Kernel function to multiply two matrices
__global__ void matrixMulA(const float* A, const float* B, float* C, int m, int n, int p) {


	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < p) {
		float sum = 0.0f;
		for (int k = 0; k < n; k++) {
			sum += A[row * n + k] * B[k * p + col];
		}
		C[row * p + col] = sum;
	}

}


int main(void) {

	// Setup for vectorAdd
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	float* h_A = (float*)malloc(size); 
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size); 

	// Initialize the input vectors
	for (int i = 0; i < numElements; ++i) {
		h_A[i] = rand() / (float)RAND_MAX; 
		h_B[i] = rand() / (float)RAND_MAX; 
	}

	float* d_A, * d_B, * d_C;
	cudaMalloc((void**)&d_A, size); 
	cudaMalloc((void**)&d_B, size); 
	cudaMalloc((void**)&d_C, size); 

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); 

	int threadsPerBlockVec = 256;
	int blocksPerGridVec = (numElements + threadsPerBlockVec - 1) / threadsPerBlockVec; 
	vectorAdd<<<blocksPerGridVec, threadsPerBlockVec>>> (d_A, d_B, d_C, numElements);   
	
	// Verify the result of vector addition
	for (int i = 0; i < numElements; ++i) {
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
				fprintf(stderr, "Result verification failed at element %d!\n", i);
				exit(EXIT_FAILURE);
		}
	}

	// Setup for matrixMul
	int m = 100, n = 100, p = 100;
	size_t sizeMatrixA = m * n * sizeof(float);
	size_t sizeMatrixB = n * p * sizeof(float);
	size_t sizeMatrixC = m * p * sizeof(float);

	float* h_MA = (float*)malloc(sizeMatrixA);
	float* h_MB = (float*)malloc(sizeMatrixB);
	float* h_MC = (float*)malloc(sizeMatrixC);

	// Initialize matrices
	for (int i = 0; i < m * n; i++) {
		h_MA[i] = rand() / (float)RAND_MAX;
	}
	for (int i = 0; i < n * p; i++) {
		h_MB[i] = rand() / (float)RAND_MAX;
	}

	// Initialize matrices
	for (int i = 0; i < m * n; i++) {
		h_MA[i] = rand() / (float)RAND_MAX;
	}
	for (int i = 0; i < n * p; i++) {
		h_MB[i] = rand() / (float)RAND_MAX; 
	}

	float* d_MA, * d_MB, * d_MC; 
	cudaMalloc(&d_MA, sizeMatrixA); 
	cudaMalloc(&d_MB, sizeMatrixB); 
	cudaMalloc(&d_MC, sizeMatrixC);

	cudaMemcpy(d_MA, h_MA, sizeMatrixA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MB, h_MB, sizeMatrixB, cudaMemcpyHostToDevice); 

	dim3 threadsPerBlock(16, 16); 
	dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y); 
	matrixMulA<<<blocksPerGrid, threadsPerBlock >>> (d_MA, d_MB, d_MC, m, n, p); 

	cudaMemcpy(h_MC, d_MC, sizeMatrixC, cudaMemcpyDeviceToHost); 

	// Free device global memory
	cudaFree(d_A); 
	cudaFree(d_B); 
	cudaFree(d_C); 
	cudaFree(d_MA); 
	cudaFree(d_MB); 
	cudaFree(d_MC); 

	// Free host memory
	free(h_A); 
	free(h_B); 
	free(h_C); 
	free(h_MA); 
	free(h_MB); 
	free(h_MC); 
	 
	printf("All Tests Done\n");
	return 0;

	/*
	// launch the kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
	
	//Setup for matrixMulA
	int m = 100, n = 100, p = 100;
	size_t sizeMatrixA = m * n * sizeof(float); 
	size_t sizeMatrixB = n * p * sizeof(float); 
	size_t sizeMatrixC = m * p * sizeof(float);

	float* h_MA, * h_MB, * h_MC; 
	float* d_MA, * d_MB, * d_MC; 

	cudaMalloc(&d_MA, sizeMatrixA);  
	cudaMalloc(&d_MB, sizeMatrixB);
	cudaMalloc(&d_MC, sizeMatrixC);

	//copy the result back to the host
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	//verify the result
	for (int i = 0; i < numElements; ++i) { 
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) { 
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE); 
		}
	}
	printf("Test PASSED\n");

	// Free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
	*/
}



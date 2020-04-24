#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

__global__ void
simple_kernel(int N, float* x, float* result) {

    // increment each element by 1
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = x[index] + 1;
}

void simple_gpu_test_function() {
	int N = 10000;
	int totalBytes = sizeof(float) * N;

	// initialize array
	float* x = (float *) malloc (totalBytes);
	float* result = (float *) malloc (totalBytes);
	for (int i=0; i<N; i++) {
		x[i] = i * 1.0;
	}

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_result;

    // allocate device memory buffers on the GPU using cudaMalloc
    cudaMalloc((void **) &device_x, N * sizeof(float));
    cudaMalloc((void **) &device_result, N * sizeof(float));

    cudaMemcpy(device_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // run kernel
    simple_kernel<<<blocks, threadsPerBlock>>>(N, device_x, device_result);
    cudaDeviceSynchronize();

    // from GPU using cudaMemcpy
    cudaMemcpy(result, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<N; i++) {
    	if (result[i] != (i*1.0)+1) {
    		printf("Incorrect result at %d %f\n", i, result[i]);
    	}
    }

    cudaFree(device_x);
    cudaFree(device_result);
    free(x);
    free(result);
}
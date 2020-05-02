#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define ZERO 0

extern float toBW(int bytes, float sec);

__global__ void
setup_zeta_leaves(int N, int* zeta, bool* leaves) {
  // compute overall index from position of thread in current block,
  // and given the block we are in
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index <= N) {
    if (leaves[index]) {
        zeta[index] = 1;
    }
  }
}

__global__ void
bfs_baseline_kernel(int N, int curr, int* levels, int* offsets,
   int* neighbours, bool* finished ) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 1 && index <= N) {
      int v = index;
      if (levels[v] == curr) {
        int num_nbr = offsets[v+1] - offsets[v];
        int offset = offsets[v];
        for(int i = 0; i < num_nbr; i++) {
          int w = neighbours[offset + i];
          if (levels[w] == ZERO) {
            *finished = false;
            levels[w] = curr + 1;
          }
        }
      }
    }
}

void
DfsCuda(int N, int M, int* offsets, int* neighbours, bool* leaves, int* p_offsets, int* parents, int** results, int* zeta) {

    int totalBytes = sizeof(int) * (6 * N + 2 * M + 8) + sizeof(bool) * (N + 1);

    // start timing
    double startTime = CycleTimer::currentSeconds();

    // compute number of blocks and threads per block
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int nnodes = N;
    int nedges = M;

    int* device_offsets;
    int* device_neighbours;
    bool* device_leaves;
    int* device_p_offsets;
    int* device_parents;
    int** device_results;
    int* device_zeta;

    //
    // allocate device memory buffers on the GPU using cudaMalloc
    //
    cudaMalloc(&device_offsets, (nnodes+2) * sizeof(int));
    cudaMalloc(&device_neighbours, nedges * sizeof(int));
    cudaMalloc(&device_leaves, (nnodes+1) * sizeof(bool));
    cudaMalloc(&device_p_offsets, (nnodes + 2) * sizeof(int));
    cudaMalloc(&device_parents, nedges * sizeof(int));
    cudaMalloc(&device_results, 3 * (nnodes + 1) * sizeof(int));
    cudaMalloc(&device_zeta, (nnodes + 1), sizeof(int));


    //
    // copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_offsets, offsets, (nnodes+2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_neighbours, neighbours, nedges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_leaves, leaves, (nnodes + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(device_p_offsets, (nnodes + 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_parents, nedges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_results, 3 * (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_zeta, (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice);


    // run kernel
    double startTime2 = CycleTimer::currentSeconds();

    // Run DFS on GPU

    // Phase 1 (Calculate zeta of nodes)

    // setup the zeta's for leaves
    setup_zeta_leaves<<<blocks, threadsPerBlock>>>(nnodes, device_zeta, device_leaves);
    cudaDeviceSynchronize();

    double endTime2 = CycleTimer::currentSeconds();

    //
    // copy result from GPU using cudaMemcpy
    //
    cudaMemcpy(zeta, device_zeta, (nnodes+1) * sizeof(int), cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    double overallDuration2 = endTime2 - startTime2;
    printf("Kernel Running Time: %.3f ms\n", 1000.f * overallDuration2);
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    // free memory buffers on the GPU
    cudaFree(device_offsets);
    cudaFree(device_neighbours);
    cudaFree(device_leaves);
    cudaFree(device_p_offsets);
    cudaFree(device_parents);
    cudaFree(device_results);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

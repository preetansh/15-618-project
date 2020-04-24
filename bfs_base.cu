#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define INF -1

__global__ void
setup_levels_kernel(int N, int* levels) {
  // compute overall index from position of thread in current block,
  // and given the block we are in
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index > 1 && index <= N) {
    levels[index] = INF;
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
        int *nbrs = &neighbours[offsets[v]];
        for(int i = 0; i < num_nbr; i++) {
          int w = nbrs[i];
          if (levels[w] == INF) {
            *finished = false;
            levels[w] = curr + 1;
          }
        }
      }
    }
}

void
BfsCuda(int N, int M, int* offsets, int* neighbours, int* levels) {

    // start timing
    double startTime = CycleTimer::currentSeconds();

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int nnodes = N;
    int nedges = M;

    int* device_offsets;
    int* device_neighbours;
    int* device_levels;

    //
    // allocate device memory buffers on the GPU using cudaMalloc
    //
    cudaMalloc(&device_offsets, (nnodes+2));
    cudaMalloc(&device_neighbours, nedges);
    cudaMalloc(&device_levels, (nndoes+1));


    //
    // copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_offsets, offsets, (nnodes+2), cudaMemcpyHostToDevice);
    cudaMemcpy(device_neighbours, neighbours, nedges, cudaMemcpyHostToDevice);

    // run kernel
    double startTime2 = CycleTimer::currentSeconds();
    // setup the levels array
    setup_levels_kernel<<<blocks, threadsPerBlock>>>(nnodes, device_levels);
    cudaThreadSynchronize();
    // run bfs_baseline_kernel
    int curr = 0;
    bool finished = true;
    do {
      finished = true;
      bfs_baseline_kernel<<<blocks, threadsPerBlock>>>(nnodes, curr++,
        device_levels, device_offsets, device_neighbours, &finished);
      cudaThreadSynchronize();
    } while(!finished);
    double endTime2 = CycleTimer::currentSeconds();

    //
    // copy result from GPU using cudaMemcpy
    //
    cudaMemcpy(levels, device_levels, (nnodes+1), cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    double overallDuration2 = endTime2 - startTime2;
    printf("Running Time: %.3f ms\n", 1000.f * overallDuration2);
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    // free memory buffers on the GPU
    cudaFree(device_offsets);
    cudaFree(device_neighbours);
    cudaFree(device_levels);
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

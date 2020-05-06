#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "bfsUtils.h"

#define ZERO 0
#define W_SIZE 32 // Virtual Warp size
#define CHUNK_SIZE 250 // Chunk size of node to process

extern float toBW(int bytes, float sec);

__global__ void
setup_levels_kernel(int N, int* levels) {
  // compute overall index from position of thread in current block,
  // and given the block we are in
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index > 1 && index <= N) {
    levels[index] = ZERO;
  }
}

// --------------------------------------------------------------------
// Warp-based BFS
// --------------------------------------------------------------------

struct warpmem_t {
    int levels[CHUNK_SIZE];
    int offsets[CHUNK_SIZE + 1];
    int scratch;
};

/*
 * memcpy_SIMD: copy from src to dst in a SIMD fashion
 */
__device__ void
memcpy_SIMD (int W_OFF, int cnt, int* dest, int* src) {
    for (int IDX = W_OFF; IDX < cnt; IDX += W_SIZE) {
        dest[IDX] = src[IDX];
    }
    __threadfence_block();
}

/*
 *
 */
__device__ void
expand_bfs_SIMD (int W_OFF, int cnt, int* edges, int* levels, int curr, int* finished) {
    for (int IDX = W_OFF; IDX < cnt; IDX += W_SIZE) {
        int v = edges[IDX];
        if (levels[v] == ZERO) {
            // use atomic CAS to set finished to 0
            atomicCAS(finished, 1, 0);
            
            levels[v] = curr + 1;
        }
    }
    __threadfence_block();
}

__global__ void
warp_bfs_kernel (int N, int curr, int* levels, int* offsets, 
    int* neighbours, int* finished) {
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int W_OFF = index % W_SIZE; // Offset of thread in current warp
    int W_ID = index / W_SIZE; // Id of this warp
    int LOCAL_W_ID = ((index % blockDim.x) / W_SIZE); // ID of this warp w.r.t this thread block 
    int num_nodes_to_process = CHUNK_SIZE;
    if ((W_ID * CHUNK_SIZE) >= N) {
        num_nodes_to_process = 0;
    } else if ( ((W_ID + 1) * CHUNK_SIZE) >= N ) {
        num_nodes_to_process = N - (W_ID * CHUNK_SIZE);
    }

    // if (W_OFF == 0) 
    //     printf("From kernel %d %d %d %d\n", W_ID, LOCAL_W_ID, index, num_nodes_to_process);

    extern __shared__ warpmem_t shared_warp_mem[];
    warpmem_t* my_warp_mem = shared_warp_mem + LOCAL_W_ID;

    // Copy work to local shared mem
    int start_node = W_ID * CHUNK_SIZE;
    memcpy_SIMD(W_OFF, num_nodes_to_process, my_warp_mem->levels, &levels[start_node]); //TODO: should not get out of bounds
    memcpy_SIMD(W_OFF, num_nodes_to_process + 1, my_warp_mem->offsets, &offsets[start_node]);

    for (int v=0; v < num_nodes_to_process; v++) {
        if (my_warp_mem->levels[v] == curr) {
            int num_nbr = my_warp_mem->offsets[v+1] - my_warp_mem->offsets[v];
            int* neigh_ptr = &neighbours[my_warp_mem->offsets[v]];
            expand_bfs_SIMD(W_OFF, num_nbr, neigh_ptr, levels, curr, finished);
        }
    }
} 

// --------------------------------------------------------------------
// Baseline BFS Kernel
// --------------------------------------------------------------------

__global__ void
bfs_baseline_kernel(int N, int curr, int* levels, int* offsets,
   int* neighbours, int* finished ) {

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
            // use atomic CAS to set finished to 0
            atomicCAS(finished, 1, 0);
            
            levels[w] = curr + 1;
          }
        }
      }
    }
}

/*
 * Perform BFS on GPU using CUDA
 * @arg BFSType: Which BFS algo to use, 0 => Baseline, 1 => Warp BFS
 */
void
BfsCuda(int N, int M, int* offsets, int* neighbours, int* globalLevels, int BFSType, int updateThreshold) {

    int totalBytes = sizeof(int) * (2 * N + M + 3);

    // start timing
    double startTime = CycleTimer::currentSeconds();

    // compute number of blocks and threads per block
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    // For warp kernel
    int warpsPerBlock = threadsPerBlock / W_SIZE;
    int numOfWarps = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const int numBlocksWarpBFS = (numOfWarps + warpsPerBlock - 1) /  warpsPerBlock;
    
    if (BFSType == 0)   printf("For baseline BFS %d %d\n", threadsPerBlock, blocks);
    if (BFSType == 1)   printf("For warp BFS %d %d %d %d\n", threadsPerBlock, warpsPerBlock, numBlocksWarpBFS, numOfWarps);
    
    int nnodes = N;
    int nedges = M;
    int* finished;
    finished = (int *) malloc(sizeof(int));
    (*finished) = 1;

    int* device_offsets;
    int* device_neighbours;
    int* device_levels;
    int* device_finished;

    //
    // allocate device memory buffers on the GPU using cudaMalloc
    //
    cudaMalloc(&device_offsets, (nnodes+2) * sizeof(int));
    cudaMalloc(&device_neighbours, nedges * sizeof(int));
    cudaMalloc(&device_levels, (nnodes+1) * sizeof(int));
    cudaMalloc(&device_finished, 1 * sizeof(int));


    //
    // copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_offsets, offsets, (nnodes+2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_neighbours, neighbours, nedges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_finished, finished, 1 * sizeof(int), cudaMemcpyHostToDevice);
    
    // run kernel
    int numBFSCalls = 0;
    double kernelTime = 0.0;
    int root = 1;
    while (root != 0) {
        // Init the levels array for this BFS
        int* levels = (int *)calloc(nnodes+1, sizeof(int));
        levels[root] = 1;
        cudaMemcpy(device_levels, levels, (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

        double strtTime = CycleTimer::currentSeconds();
        
        // run bfs_baseline_kernel for this root
        int curr = 1;
        do {
            *finished = 1;
            cudaMemcpy(device_finished, finished, 1 * sizeof(int), cudaMemcpyHostToDevice);
          
            if (BFSType == 0) {
                // call baseline kernel
                bfs_baseline_kernel<<<blocks, threadsPerBlock>>>(nnodes, curr++,
                device_levels, device_offsets, device_neighbours, device_finished);
            } else {
                // call warp BFS kernel
                warp_bfs_kernel<<<numBlocksWarpBFS, threadsPerBlock, warpsPerBlock * sizeof(warpmem_t)>>>(nnodes, curr++,
                device_levels, device_offsets, device_neighbours, device_finished);
            }
          
          
            cudaDeviceSynchronize();
            cudaMemcpy(finished, device_finished, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        } while(!((*finished) == 1));
        
        double enTime = CycleTimer::currentSeconds();
        
        //
        // copy result from GPU using cudaMemcpy
        //
        cudaMemcpy(levels, device_levels, (nnodes+1) * sizeof(int), cudaMemcpyDeviceToHost);

        // move visited to visitedGlobal and get next root
        std::pair<int, int> update = updateGlobalVisited(globalLevels, levels, nnodes);
        if (update.second >= updateThreshold) {
            // Add to total time
            kernelTime += enTime - strtTime;
            numBFSCalls++;
        }
        root = update.first;

        free(levels);
    }

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    double avgKernelTime = kernelTime / numBFSCalls;
    printf("Kernel Running Time: %.3f ms\n", 1000.f * avgKernelTime);
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

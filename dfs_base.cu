#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define ZERO 0

extern float toBW(int bytes, float sec);

__global__ void
zero_edge_weights(int M, int* edge_weights) {
  // compute overall index from position of thread in current block,
  // and given the block we are in
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < M) {
    edge_weights[index] = 0;
  }
}

__global__ void
setup_zeta_leaves(int N, int* zeta, bool* leaves, bool* q_queue, int* child_counter) {
  // compute overall index from position of thread in current block,
  // and given the block we are in
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index <= N) {
    if (leaves[index]) {
        zeta[index] = 1;
        q_queue[index] = true;
    }
    child_counter[index] = 0;
  }
}

__global__ void
propagate_zeta(int N, int* zeta, int* edge_weights, bool* q_queue, bool* c_queue, int* offsets, 
    int* neighbours, int* p_offsets, int* parents, int* child_to_parent) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index <= N) {
        if (q_queue[index]) {
            int child_offset = p_offsets[index];
            int num_parents = p_offsets[index + 1] - child_offset;
            // TODO : Parallelizing this for loop might help
            for (int i = 0; i < num_parents; i++) {
                int j = child_to_parent[child_offset + i];
                edge_weights[j] = zeta[index];
            }
        }
    }
}

__global__ void
calculate_parent_zeta(int N, int* zeta, int* edge_weights, bool* q_queue, bool* c_queue, int* offsets, 
    int* neighbours, int* p_offsets, int* parents, int* child_to_parent) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index <= N && zeta[index] == 0) {
        int n_offset = offsets[index];
        int n_children = offsets[index + 1] - n_offset;

        bool flag = true;

        for (int i = 0; i < n_children; i++) {
            if (edge_weights[n_offset + i] == 0) {
                flag = false;
                break;
            }
        }

        if(flag) {
            int prefix_sum = 1;
            for (int i = 0; i < n_children; i++) {
                int temp = edge_weights[n_offset + i];
                edge_weights[n_offset + i] = prefix_sum;
                prefix_sum += temp;
            }
            zeta[index] = prefix_sum;
            c_queue[index] = true;
        }
    }
}

__global__ void
exchange_c_q(int N, bool* q_queue, bool* c_queue) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index <= N) {
        q_queue[index] = c_queue[index];
        c_queue[index] = 0;
    }
}

void
DfsCuda(int N, int M, int* offsets, int* neighbours, bool* leaves, int* p_offsets, 
    int* parents, int* child_to_parent, int** results, int* zeta) {

    int totalBytes = sizeof(int) * (6 * N + 3 * M + 8) + sizeof(bool) * (N + 1);

    // start timing
    double startTime = CycleTimer::currentSeconds();

    // compute number of blocks and threads per block
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int blocks_edges = (M + threadsPerBlock - 1) / threadsPerBlock;
    int nnodes = N;
    int nedges = M;

    int* device_offsets;
    int* device_neighbours;
    bool* device_leaves;
    int* device_p_offsets;
    int* device_parents;
    int** device_results;
    int* device_zeta;
    int* device_child_to_parent;

    int* device_edge_weights;
    int* child_counter;
    bool* c_queue;
    bool* q_queue;
    //
    // allocate device memory buffers on the GPU using cudaMalloc
    //
    cudaMalloc(&device_offsets, (nnodes+2) * sizeof(int));
    cudaMalloc(&device_neighbours, nedges * sizeof(int));
    cudaMalloc(&device_leaves, (nnodes+1) * sizeof(bool));
    cudaMalloc(&device_p_offsets, (nnodes + 2) * sizeof(int));
    cudaMalloc(&device_parents, nedges * sizeof(int));
    cudaMalloc(&device_results, 3 * (nnodes + 1) * sizeof(int));
    cudaMalloc(&device_zeta, (nnodes + 1) * sizeof(int));
    cudaMalloc(&device_child_to_parent, (nedges) * sizeof(int));

    cudaMalloc(&device_edge_weights, nedges * sizeof(int));
    cudaMalloc(&c_queue, (nnodes + 1) * sizeof(bool));
    cudaMalloc(&q_queue, (nnodes + 1) * sizeof(bool));
    cudaMalloc(&child_counter, (nnodes + 1)*sizeof(int));

    cudaMemset(c_queue, false, (nnodes + 1) * sizeof(bool));
    cudaMemset(q_queue, false, (nnodes + 1) * sizeof(bool));

    //
    // copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_offsets, offsets, (nnodes+2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_neighbours, neighbours, nedges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_leaves, leaves, (nnodes + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(device_p_offsets, p_offsets, (nnodes + 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_parents, parents, nedges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_results, results, 3 * (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_zeta, zeta, (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_child_to_parent, child_to_parent, (nedges) * sizeof(int), cudaMemcpyHostToDevice);


    // run kernel
    double startTime2 = CycleTimer::currentSeconds();

    // Run DFS on GPU

    // Phase 1 (Calculate zeta of nodes)

    // setup the zeta's for leaves and initialize q with the leaves
    zero_edge_weights<<<blocks_edges, threadsPerBlock>>>(nedges, device_edge_weights);
    cudaDeviceSynchronize();
    setup_zeta_leaves<<<blocks, threadsPerBlock>>>(nnodes, device_zeta, device_leaves, q_queue, child_counter);
    cudaDeviceSynchronize();
    int* edge_weights = (int *) malloc(sizeof(int) * (nedges));

    // calculate edge weights
    while(true) {
        int zeta_of_zero = 0;

        propagate_zeta<<<blocks, threadsPerBlock>>>(nnodes, device_zeta, device_edge_weights, q_queue, 
            c_queue, device_offsets, device_neighbours, device_p_offsets, device_parents, device_child_to_parent);
        cudaDeviceSynchronize();

        calculate_parent_zeta<<<blocks, threadsPerBlock>>>(nnodes, device_zeta, device_edge_weights, q_queue, 
            c_queue, device_offsets, device_neighbours, device_p_offsets, device_parents, device_child_to_parent);
        cudaDeviceSynchronize();

        exchange_c_q<<<blocks, threadsPerBlock>>>(nnodes, q_queue, c_queue);
        cudaDeviceSynchronize();

        cudaMemcpy(&zeta_of_zero, device_zeta, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        if (zeta_of_zero) {
            break;
        }
    }

    cudaDeviceSynchronize();

    double endTime2 = CycleTimer::currentSeconds();

    //
    // copy result from GPU using cudaMemcpy
    //
    cudaMemcpy(zeta, device_zeta, (nnodes+1) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(edge_weights, device_edge_weights, nedges * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i<=nnodes; i++) {
        int offset = offsets[i];
        for (int j = 0; j < (offsets[i+1] - offsets[i]); j++) {
            int child = neighbours[offset + j];
            std::cout << i << " - " << child << " and edge weight " << edge_weights[offset + j] << "\n";
        }
    }
    free(edge_weights);

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
    cudaFree(device_zeta);
    cudaFree(device_child_to_parent);

    cudaFree(device_edge_weights);
    cudaFree(child_counter);
    cudaFree(c_queue);
    cudaFree(q_queue);
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

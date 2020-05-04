#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define ZERO 0

extern float toBW(int bytes, float sec);

__global__ void
zero_edge_weights(int M, long long* edge_weights) {
  // compute overall index from position of thread in current block,
  // and given the block we are in
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < M) {
    edge_weights[index] = 0;
  }
}

__global__ void
setup_zeta_leaves(int N, long long* zeta, bool* leaves, bool* q_queue) {
  // compute overall index from position of thread in current block,
  // and given the block we are in
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index <= N) {
    if (leaves[index]) {
        zeta[index] = 1;
        q_queue[index] = true;
    }
  }
}

__global__ void
propagate_zeta(int N, long long* zeta, long long* edge_weights, bool* q_queue, bool* c_queue, int* offsets, 
    int* neighbours, int* p_offsets, int* parents, int* child_to_parent) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index <= N) {
        if (q_queue[index]) {
            // printf("propagating zeta for %d\n", index);
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
calculate_parent_zeta(int N, long long* zeta, long long* edge_weights, bool* q_queue, bool* c_queue, int* offsets, 
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
            long long prefix_sum = 1;
            for (int i = 0; i < n_children; i++) {
                long long temp = edge_weights[n_offset + i];
                edge_weights[n_offset + i] = prefix_sum;
                if (prefix_sum < 0) {
                    printf("Edge number %d\n", n_offset + i);
                }
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

__global__ void
zero_child_costs(int nedges, long long* child_costs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nedges) {
        child_costs[index] = 0;
    }
}

__global__ void
setup_phase_2(int N, long long* cost, bool* q_queue, bool* c_queue) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index <= N) {
        if(index == 0) {
            cost[index] = 0;
            q_queue[index] = true;
        }
        else {
            cost[index] = LLONG_MAX; // TODO: Change this value
            q_queue[index] = false;
        }
        c_queue[index] = false;
    }
}


__global__ void
calculate_cost(int nnodes, long long* cost, bool *q_queue, bool *c_queue, int* offsets, 
    int* neighbours, long long* edge_weights, int* parent_to_child, long long* child_costs, bool* explored) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= nnodes) {
        if (q_queue[index]) {
            // printf("Cost index %d\n", index);
            int c_offset = offsets[index];
            int num_children = offsets[index + 1] - c_offset;
            for (int i = 0; i < num_children; i++) {
                long long new_cost = cost[index] + edge_weights[i + c_offset];
                int parent_to_child_index = parent_to_child[c_offset + i];
                child_costs[parent_to_child_index] = new_cost;
            }
            explored[index] = true;
        }
    }
}

__global__ void
construct_c_queue(int nnodes, bool *c_queue, int *p_offsets, int* parents,
    long long* child_costs, int* results, long long* cost, bool* explored) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= nnodes && (explored[index] == false)) {
        int p_start = p_offsets[index];
        int num_parents = p_offsets[index + 1] - p_start;
        bool flag = true;
        long long local_cost = LLONG_MAX;
        int local_parent = -1;
        for (int i = 0; i < num_parents; i++) {
            if (explored[parents[i+p_start]] == false) {
                flag = false;
                break;
            } 
            else {
                if (child_costs[p_start + i] < local_cost) {
                    local_cost = child_costs[p_start + i];
                    local_parent = parents[i + p_start];
                }
            }
        }
        if (flag) {
            // printf("C_queue entering : %d\n", index);
            c_queue[index] = true;
            cost[index] = local_cost;
            results[index] = local_parent;
        }
    }
}

__global__ void
check_all_explored(int nnodes, bool* all_explored, bool* explored) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= nnodes) {
        if (explored[index] == false) {
            *all_explored = false;
        }
    }
}

void
DfsCuda(int N, int M, int* offsets, int* neighbours, bool* leaves, int* p_offsets, 
    int* parents, int* child_to_parent, int* parent_to_child, int** results, long long* zeta) {

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
    int* device_discovery;
    int* device_result_parents;
    int* device_finish;
    long long* device_zeta;
    int* device_child_to_parent;
    int* device_parent_to_child;

    long long* device_edge_weights;
    long long* device_cost;
    bool* device_c_queue;
    bool* device_q_queue;
    bool* device_explored_2;
    long long* device_child_costs;
    //
    // allocate device memory buffers on the GPU using cudaMalloc
    //
    cudaMalloc(&device_offsets, (nnodes+2) * sizeof(int));
    cudaMalloc(&device_neighbours, nedges * sizeof(int));
    cudaMalloc(&device_leaves, (nnodes+1) * sizeof(bool));
    cudaMalloc(&device_p_offsets, (nnodes + 2) * sizeof(int));
    cudaMalloc(&device_parents, nedges * sizeof(int));
    cudaMalloc(&device_discovery, (nnodes + 1) * sizeof(int));
    cudaMalloc(&device_result_parents, (nnodes + 1) * sizeof(int));
    cudaMalloc(&device_finish, (nnodes + 1) * sizeof(int));
    cudaMalloc(&device_zeta, (nnodes + 1) * sizeof(long long));
    cudaMalloc(&device_child_to_parent, (nedges) * sizeof(int));
    cudaMalloc(&device_parent_to_child, (nedges) * sizeof(int));

    cudaMalloc(&device_edge_weights, nedges * sizeof(long long));
    cudaMalloc(&device_child_costs, nedges * sizeof(long long));
    cudaMalloc(&device_c_queue, (nnodes + 1) * sizeof(bool));
    cudaMalloc(&device_q_queue, (nnodes + 1) * sizeof(bool));
    cudaMalloc(&device_cost, (nnodes + 1)*sizeof(long long));
    cudaMalloc(&device_explored_2, (nnodes + 1)*sizeof(int));

    cudaMemset(device_c_queue, false, (nnodes + 1) * sizeof(bool));
    cudaMemset(device_q_queue, false, (nnodes + 1) * sizeof(bool));
    cudaMemset(device_explored_2, false, (nnodes + 1) * sizeof(bool));

    //
    // copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_offsets, offsets, (nnodes+2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_neighbours, neighbours, nedges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_leaves, leaves, (nnodes + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(device_p_offsets, p_offsets, (nnodes + 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_parents, parents, nedges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_discovery, results[0], (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result_parents, results[1], (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_finish, results[2], (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_zeta, zeta, (nnodes + 1) * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(device_child_to_parent, child_to_parent, (nedges) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_parent_to_child, parent_to_child, (nedges) * sizeof(int), cudaMemcpyHostToDevice);


    // run kernel
    double startTime2 = CycleTimer::currentSeconds();

    // Run DFS on GPU
    std::cout << "Starting GPU" << "\n";
    // Phase 1 (Calculate zeta of nodes)

    // setup the zeta's for leaves and initialize q with the leaves
    zero_edge_weights<<<blocks_edges, threadsPerBlock>>>(nedges, device_edge_weights);
    cudaDeviceSynchronize();
    setup_zeta_leaves<<<blocks, threadsPerBlock>>>(nnodes, device_zeta, device_leaves, device_q_queue);
    cudaDeviceSynchronize();
    long long* edge_weights = (long long *) malloc(sizeof(long long) * (nedges));

    // calculate edge weights
    while(true) {
        long long zeta_of_zero = 0;

        propagate_zeta<<<blocks, threadsPerBlock>>>(nnodes, device_zeta, device_edge_weights, device_q_queue, 
            device_c_queue, device_offsets, device_neighbours, device_p_offsets, device_parents, device_child_to_parent);
        cudaDeviceSynchronize();

        // std::cout << "__________" << "\n";

        calculate_parent_zeta<<<blocks, threadsPerBlock>>>(nnodes, device_zeta, device_edge_weights, device_q_queue, 
            device_c_queue, device_offsets, device_neighbours, device_p_offsets, device_parents, device_child_to_parent);
        cudaDeviceSynchronize();

        // std::cout << "************" << "\n";

        exchange_c_q<<<blocks, threadsPerBlock>>>(nnodes, device_q_queue, device_c_queue);
        cudaDeviceSynchronize();

        // stopping condition
        cudaMemcpy(&zeta_of_zero, device_zeta, 1 * sizeof(long long), cudaMemcpyDeviceToHost);
        if (zeta_of_zero) {
            break;
        }
    }

    std::cout << "Calculated edge weights" << "\n";

    // cudaMemcpy(edge_weights, device_edge_weights, nedges * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i<=nnodes; i++) {
    //     int offset = offsets[i];
    //     for (int j = 0; j < (offsets[i+1] - offsets[i]); j++) {
    //         int child = neighbours[offset + j];
    //         if (edge_weights[offset + j] <= 0) {
    //             std::cout << i << " - " << child << " and edge weight " << edge_weights[offset + j] << "\n";
    //         }
    //     }
    // }

    // Phase 2 (Calculate cost of paths and parents)

    // setup the cost and the queues
    zero_child_costs<<<blocks_edges, threadsPerBlock>>>(nedges, device_child_costs);
    cudaDeviceSynchronize();
    setup_phase_2<<<blocks, threadsPerBlock>>>(nnodes, device_cost, device_q_queue, device_c_queue);

    bool* device_all_explored;
    cudaMalloc(&device_all_explored, 1 * sizeof(bool));

    while(true) {
        bool all_explored = true;             

        calculate_cost<<<blocks, threadsPerBlock>>>(nnodes, device_cost, device_q_queue, device_c_queue, device_offsets, 
            device_neighbours, device_edge_weights, device_parent_to_child, device_child_costs, device_explored_2);
        cudaDeviceSynchronize();

        construct_c_queue<<<blocks, threadsPerBlock>>>(nnodes, device_c_queue, device_p_offsets, 
            device_parents, device_child_costs, device_result_parents, device_cost, device_explored_2);
        cudaDeviceSynchronize();

        exchange_c_q<<<blocks, threadsPerBlock>>>(nnodes, device_q_queue, device_c_queue);
        cudaDeviceSynchronize();

        cudaMemcpy(device_all_explored, &all_explored, 1 * sizeof(bool), cudaMemcpyHostToDevice);
        check_all_explored<<<blocks, threadsPerBlock>>>(nnodes, device_all_explored, device_explored_2);
        cudaDeviceSynchronize();
        cudaMemcpy(&all_explored, device_all_explored, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
        if (all_explored) {
            break;
        }
    } 
    cudaFree(device_all_explored);

    cudaDeviceSynchronize();

    double endTime2 = CycleTimer::currentSeconds();

    std::cout << "Calculated parents" << "\n";

    //
    // copy result from GPU using cudaMemcpy
    //
    cudaMemcpy(zeta, device_zeta, (nnodes+1) * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(results[1], device_result_parents, (nnodes+1) * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(results[0], device_cost, (nnodes+1) * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i <= nnodes; i++) {
    //     std::cout << "Parent of " << i << " is : " << results[1][i] << " and cost is : " << results[0][i] << "\n";
    // }

    // cudaMemcpy(edge_weights, device_edge_weights, nedges * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i<=nnodes; i++) {
    //     int offset = offsets[i];
    //     for (int j = 0; j < (offsets[i+1] - offsets[i]); j++) {
    //         int child = neighbours[offset + j];
    //         std::cout << i << " - " << child << " and edge weight " << edge_weights[offset + j] << "\n";
    //     }
    // }
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
    cudaFree(device_discovery);
    cudaFree(device_result_parents);
    cudaFree(device_finish);
    cudaFree(device_zeta);
    cudaFree(device_child_to_parent);
    cudaFree(device_parent_to_child);

    cudaFree(device_edge_weights);
    cudaFree(device_child_costs);
    cudaFree(device_cost);
    cudaFree(device_c_queue);
    cudaFree(device_q_queue);
    cudaFree(device_explored_2);
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

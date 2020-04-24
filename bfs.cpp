#include <iostream>
#include <stdlib.h>
#include <queue>

#include "graph.h"
#include "CycleTimer.h"

void BfsCuda(int N, int M, int* offsets, int* neighbours, int* levels);
void printCudaInfo();

// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

void BFSSeq (Graph* g, int* visited) {
	std::queue<int> pending;

	int* offsets = g->GetOffsets();
	int* neighbors = g->GetNeighbours();

	// Add starting node
	visited[1] = 1;
	pending.push(1);

	// Start BFS loop
	while (!pending.empty()) {
		// get next node
		int curr = pending.front();
		pending.pop();
		int curr_level = visited[curr];

		// iterate over its neighbors
		int neighbour_start = offsets[curr];
		int neighbour_end = offsets[curr+1];
		for (int i=0; i < (neighbour_end - neighbour_start); i++) {
			int neigh = neighbors[neighbour_start + i];
			// if not visited
			if (visited[neigh] == 0) {
				visited[neigh] = curr_level + 1;
				pending.push(neigh);
			}
		}
	}

	return;
}

int main()
{
	Graph* g = new Graph();

	char* filename = "data/ca2010/ca2010.mtx";
	std::cout << "Reading " << filename << "\n";

	double startTime = CycleTimer::currentSeconds();
	g->ReadGraph(filename, true);
	double endTime = CycleTimer::currentSeconds();

	std::cout << "Graph Read. Found " << g->GetNodes() << " nodes and " << g->GetEdges() << " edges" << std::endl;
	std::cout << "Read completed in " << (endTime - startTime) << std::endl;

	int* neighbours = g->GetNeighbours();
	int* offsets = g->GetOffsets();
	int n = g->GetNodes();
	int m = g->GetEdges();
	int* levels = (int *) calloc((n + 1), sizeof(int));
	int* visited = (int *)calloc(g->GetNodes()+1, sizeof(int));

	startTime = CycleTimer::currentSeconds();
	BFSSeq(g, visited);
	endTime = CycleTimer::currentSeconds();
	std::cout << "Sequential Single Thread BFS completed in " << (endTime - startTime) << std::endl;

	// check the state of the gpu
	printCudaInfo();

	// run the main cuda program (timing starts inside)
    BfsCuda(n, m, offsets, neighbours, levels);

	// checking GPU output
	bool correct = true;
	for (int i = 1; i <= n; i++) {
		if (visited[i] != levels[i]) {
			std::cout << "Sequential and Parallel conflict at " << i << std::endl;
			std::cout << "Sequential level  " << visited[i] << std::endl;
			std::cout << "Parallel level " << levels[i] << std::endl;
			correct = false;
			break;
		}
	} 

	std::cout << "Correctness : " << correct << "\n";
	

	free(visited);
	free(levels);
	g->FreeGraph();
	delete(g);

	return 0;
}
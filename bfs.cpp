#include <iostream>
#include <stdlib.h>
#include <queue>

#include "graph.h"
#include "CycleTimer.h"

void BFSSeq (Graph* g, int* visited) {
	int nnodes = g->GetNodes();
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

	double startTime = CycleTimer::currentSeconds();
	g->ReadGraph("road_usa.mtx");
	double endTime = CycleTimer::currentSeconds();

	std::cout << "Graph Read. Found " << g->GetNodes() << " nodes and " << g->GetEdges() << " edges" << std::endl;
	std::cout << "Read completed in " << (endTime - startTime) << std::endl;

	for (int i = 0; i < 10; i++) {
		startTime = CycleTimer::currentSeconds();
		int* visited = (int *)calloc(g->GetNodes()+1, sizeof(int));
		BFSSeq(g, visited);
		endTime = CycleTimer::currentSeconds();
		std::cout << "BFS completed in " << (endTime - startTime) << std::endl;

		for (int i=1; i<=g->GetNodes(); i++) {
			if (visited[i] == 0) {
				std::cout << i << ": " << visited[i] << std::endl;
			}
		}
		free(visited);
	}

	g->FreeGraph();
	delete(g);

	return 0;
}
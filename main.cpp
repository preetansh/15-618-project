#include "graph.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <string.h>
#include <map>

void BfsCuda(int N, int M, int* offsets, int* neighbours, int* levels);
void printCudaInfo();

// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


int main() {
	Graph* g = new Graph();

	// read and initialize the graph, not a timed operation
	g->ReadGraph("sample.mtx");
	std::cout << g->GetNodes() << "\n";
	std::cout << g->GetEdges() << "\n";

	int* neighbours = g->GetNeighbours();
	int* offsets = g->GetOffsets();
<<<<<<< HEAD
	int n = g->GetNodes();
	int m = g->GetEdges();
	int* levels = (int *) calloc((n + 1), sizeof(int));
=======
>>>>>>> ee56094faf41e44699d5da01bbb27fe7a186b52d

	// check the state of the gpu
	printCudaInfo();

	// run the main cuda program (timing starts inside)
    BfsCuda(n, m, offsets, neighbours, levels);

    for (int i = 1; i <= n; i+=1000) {
    	std::cout << "Level for node " << i << " : " << levels[i] << "\n";
    }
<<<<<<< HEAD

	
	free(levels);
=======
>>>>>>> ee56094faf41e44699d5da01bbb27fe7a186b52d
    g->FreeGraph();
    delete(g);
}
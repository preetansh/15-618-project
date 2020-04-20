#include "graph.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <string.h>
#include <map>

int main() {
	Graph* g = new Graph();
	g->ReadGraph("sample.mtx");
	std::cout << g->GetNodes() << "\n";
	std::cout << g->GetEdges() << "\n";

	std::vector<int> neighbours = g->GetNeighbours();
	std::vector<int> offsets = g->GetOffsets();

	for (int i = 1; i <= g->GetNodes(); i++) {
    	std::cout << "Offset for node " << i << " is " << offsets[i] << "\n";
    	for (int j = 0; j < (offsets[i+1] - offsets[i]); j++) {
    		std::cout << "Neighbour for node " << i << " is " << neighbours[offsets[i]+j] << "\n";
    	}
    }
}
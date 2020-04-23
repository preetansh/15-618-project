#include "graph.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <map>

int d = 0;
int f = 0;
std::map<int, int> visited;

void dfs(Graph* g, int root, int** results) {
	visited[root] = 1;
	results[0][root] = d;
	d++;

	int* offsets = g->GetOffsets();
	int* neighbours = g->GetNeighbours();
	int offset = offsets[root];
	int num_neighbours = offsets[root+1] - offset;
	
	for (int i = 0; i < num_neighbours; i++) {
		int child = neighbours[offset + i];

		if (visited.count(child) == 0) {
			results[1][child] = root;
			dfs(g, child, results);
		}
	}
	results[2][root] = f; 
	f++;
} 

int main() {
	Graph* g = new Graph();
	g->ReadGraph("sample.mtx");
	int nnodes = g->GetNodes();

	int** results = (int **) malloc(sizeof(int *) * 3);
	int* pre_order = (int *) calloc((nnodes + 1), sizeof(int)); // calloc assures 0 at each position
	int* parent = (int *) calloc((nnodes + 1), sizeof(int));
	int* post_order = (int *) calloc((nnodes + 1), sizeof(int));

	results[0] = pre_order;
	results[1] = parent;
	results[2] = post_order;

	for (int i = 0; i < (nnodes + 1); i++) {
		pre_order[i] = -1;
		parent[i] = -1;
		post_order[i] = -1;
	}

	dfs(g, 1, results);

	for (int i = 0; i < 3; i++) {
		for (int j = 1; j <= nnodes; j++) {
				std::cout << results[i][j] << " ";
		}
		std::cout << "\n";
	} 

	free(pre_order);
	free(parent);
	free(post_order);
	free(results);
	g->FreeGraph();
	delete(g);
}
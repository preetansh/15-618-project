#include "graph.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <map>
#include <stack>
#include "CycleTimer.h"

int d = 0;
int f = 0;

void recursive_dfs(Graph* g, int root, int** results) {
	results[0][root] = d;
	d++;

	int* offsets = g->GetOffsets();
	int* neighbours = g->GetNeighbours();
	int offset = offsets[root];
	int num_neighbours = offsets[root+1] - offset;
	
	for (int i = 0; i < num_neighbours; i++) {
		int child = neighbours[offset + i];

		if (results[0][child] == -1) { // not set in pre-order means not visited
			results[1][child] = root;
			recursive_dfs(g, child, results);
		}
	}
	results[2][root] = f; 
	f++;
} 

void iterative_dfs(Graph *g, int root, int** results) {
	int* offsets = g->GetOffsets();
	int* neighbours = g->GetNeighbours();
	std::stack<int> nodes_left;
	nodes_left.push(root);
	while(!(nodes_left.empty())) {
		int v = nodes_left.top();
		if (results[2][v] == -1) { // v not finished
			if (results[0][v] == -1) { // not discovered
				results[0][v] = d;
				d++;
				int offset = offsets[v];
				int num_neighbours = offsets[v+1] - offset;
				for (int i = num_neighbours-1; i >= 0; i--) {
					int child = neighbours[offset + i];
					if (results[2][child] == -1) { // child should not be finished
						results[1][child] = v; // set child as parent
						nodes_left.push(child);
					}
				}
			}
			else {
				results[2][v] = f;
				f++;
				nodes_left.pop();
			}
		}
		else {
			nodes_left.pop();
		}
	}
}

void dfs(Graph *g, int** results) {
	for (int i = 1; i <= g->GetNodes(); i++) {
		if (results[2][i] == -1) {
			iterative_dfs(g, i, results);
		} 
	}
}

int main() {
	Graph* g = new Graph();
	g->ReadGraph("data/coPaper.mtx", false, true);
	std::cout << "Read the graph" << "\n";

	int nnodes = g->GetNodes();

	int* offsets = g->GetOffsets();
	int* neighbours = g->GetNeighbours();
	for (int i = 1; i<=nnodes; i++) {
		int offset = offsets[i];
		for (int j = 0; j < (offsets[i+1] - offsets[i]); j++) {
			int child = neighbours[offset + j];
		}
	}

	double startTime = CycleTimer::currentSeconds();

	int** results = (int **) malloc(sizeof(int *) * 3);
	int* pre_order = (int *) calloc((nnodes + 1), sizeof(int)); // calloc assures 0 at each position
	int* parent = (int *) calloc((nnodes + 1), sizeof(int));
	int* post_order = (int *) calloc((nnodes + 1), sizeof(int));

	for (int i = 0; i < (nnodes + 1); i++) {
		pre_order[i] = -1;
		parent[i] = -1;
		post_order[i] = -1;
	}

	results[0] = pre_order;
	results[1] = parent;
	results[2] = post_order;


	dfs(g, results);

	double endTime = CycleTimer::currentSeconds();


	std::cout << "Time taken for dfs is " << (endTime - startTime) << "\n";
	// print results
	// for (int i = 0; i < 3; i++) {
	// 	for (int j = 1; j <= nnodes; j++) {
	// 			std::cout << j << " - " << results[i][j] << "\n";
	// 	}
	// 	std::cout << "Changing of result" << "\n";
	// } 

	free(pre_order);
	free(parent);
	free(post_order);
	free(results);
	g->FreeGraph();
	delete(g);
}

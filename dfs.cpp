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

void printGraphInfo(Graph* g) {
	std::cout << "Printing Graph Info" << "\n";

	int nnodes = g->GetNodes();

	int* offsets = g->GetOffsets();
	int* neighbours = g->GetNeighbours();
	for (int i = 0; i<=nnodes; i++) {
		int offset = offsets[i];
		for (int j = 0; j < (offsets[i+1] - offsets[i]); j++) {
			int child = neighbours[offset + j];
			std::cout << i << " - " << child << "\n";
		}
	}
	std::cout << "Next : Parents" << "\n";
	int* p_offsets = g->GetParentOffsets();
	int* p_parents = g->GetParents();
	for (int i = 0; i<=nnodes; i++) {
		int offset = p_offsets[i];
		for (int j = 0; j < (p_offsets[i+1] - p_offsets[i]); j++) {
			int child = p_parents[offset + j];
			std::cout << i << " - " << child << "\n";
		}
	}

	std::cout << "Next : Roots" << "\n";
	bool* r = g->GetRoots();
	for (int i = 0; i <= nnodes; i++) {
		std::cout << i << " - " << r[i] << "\n";
	}

	std::cout << "Next : Leaves" << "\n";
	bool* l = g->GetLeaves();
	for (int i = 0; i <= nnodes; i++) {
		std::cout << i << " - " << l[i] << "\n";
	}
}

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
	iterative_dfs(g, 0, results);
}

void printDFSResults(int** results, int nnodes) {
	std::cout << "Printing DFS Results" << "\n";
	std::cout << "Discovery time" << "\n";
	for(int i = 0; i <= nnodes; i++) {
		std::cout << i << " - " << results[0][i] << "\n";
	}

	std::cout << "Parents" << "\n";
	for(int i = 0; i <= nnodes; i++) {
		std::cout << i << " - " << results[1][i] << "\n";
	}

	std::cout << "Finish time" << "\n";
	for(int i = 0; i <= nnodes; i++) {
		std::cout << i << " - " << results[2][i] << "\n";
	}
}

int main() {
	Graph* g = new Graph();
	g->ReadDFSGraph("data/sample2.mtx", false);
	int nnodes = g->GetNodes();

	std::cout << "Read Graph for DFS" << "\n";
	// printGraphInfo(g); // uncomment to print the info of graph

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

	// printDFSResults(results, nnodes); // uncomment to print the dfs results

	free(pre_order);
	free(parent);
	free(post_order);
	free(results);
	g->FreeGraph();
	delete(g);
}

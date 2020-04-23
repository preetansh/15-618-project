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

void dfs(Graph& g, int root, std::vector<std::vector<int> >& results) {
	visited[root] = 1;
	results[0][root] = d;
	d++;

	std::vector<int> offsets = g.GetOffsets();
	int offset = offsets[root];
	int num_neighbors = offsets[root+1] - offset;
	
	for (int i = 0; i < num_neighbors; i++) {
		int child = g.GetNeighbours()[offset + i];

		if (visited.count(child) == 0) {
			results[1][child] = root;
			dfs(g, child, results);
		}
	}
	results[2][root] = f; 
	f++;
} 

int main() {
	Graph g;
	g.ReadGraph("sample.mtx");
	int nnodes = g.GetNodes();

	std::vector<std::vector<int> > results;
	std::vector<int> pre_order(nnodes+1, -1);
	std::vector<int> parent(nnodes+1, -1);
	std::vector<int> post_order(nnodes+1, -1);
	results.push_back(pre_order);
	results.push_back(parent);
	results.push_back(post_order);

	dfs(g, 1, results);

	for (int i = 0; i < 3; i++) {
		for (int j = 1; j <= nnodes; j++) {
				std::cout << results[i][j] << " ";
		}
		std::cout << "\n";
	} 
}
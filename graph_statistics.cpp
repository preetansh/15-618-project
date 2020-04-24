#include "graph.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <math.h>

std::vector<double> GetStatistics(Graph* g) {
	int n = g->GetNodes();
	int m = g->GetEdges();
	int* offsets = g->GetOffsets();

	std::vector<double> degree;
	for (int i = 1; i <= n; i++) {
		int num_edges = offsets[i+1] - offsets[i];
		degree.push_back(num_edges); 
	}

	std::sort(degree.begin(), degree.end());

	double min_degree = degree[0];
	double max_degree = degree[n-1];
	double median_degree = degree[n/2];
	double mean_degree = std::accumulate(degree.begin(), degree.end(), 0.0)/degree.size();
	double sq_sum = std::inner_product(degree.begin(), degree.end(), degree.begin(), 0.0);
	double stdev_degree = std::sqrt(sq_sum / degree.size() - mean_degree * mean_degree);

	std::vector<double> statistics;
	statistics.push_back(min_degree);
	statistics.push_back(max_degree);
	statistics.push_back(median_degree);
	statistics.push_back(mean_degree);
	statistics.push_back(stdev_degree);
	return statistics;
}

int main() {
	Graph* g = new Graph();
	g->ReadGraph("data/venturiLevel3.mtx");

	std::cout << "Graph read" << "\n";

	int nnodes = g->GetNodes();
	int nedges = g->GetEdges();

	std::vector<double> stats = GetStatistics(g);

	std::cout << "Nodes : " << nnodes << " Edges : " << nedges << "\n";
	std::cout << "Min degree : " << stats[0] << " Max degree : " << stats[1] << " Median degree : " << stats[2] << "\n";
	std::cout << "Mean degree : " << stats[3] << " Stdev degree : " << stats[4] << "\n";

	
	g->FreeGraph();
	delete(g);
}
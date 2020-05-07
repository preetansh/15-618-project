#include <iostream>
#include <stdlib.h>
#include <queue>
#include <map>

#include "graph.h"
#include "CycleTimer.h"

#define UPDATE_THRESHOLD 10 // Min. number of level updates this BFS to be considered

void BfsCuda(int N, int M, int* offsets, int* neighbours, int* levels, int BFSType, int updateThreshold);
void printCudaInfo();
std::pair<int, int> updateGlobalVisited(int* visitedGlobal, int* visited, int n);

// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

void BFSSeq (Graph* g, int root, int* visited) {
	std::queue<int> pending;

	int* offsets = g->GetOffsets();
	int* neighbors = g->GetNeighbours();

	// Add starting node
	visited[root] = 1;
	pending.push(root);

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

/*
 *
 * File locations: 
  * coPaper: "/afs/andrew.cmu.edu/usr21/preetang/private/courses/15618/15-618-project/data/coPapersDBLP/coPapersDBLP.mtx" <unweighted>
 * road_usa: "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/road_usa/road_usa.mtx" <unweighted>
 * hugebubbles-00020: "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/hugebubbles-00020/hugebubbles-00020.mtx" <unweighted> 
 * ca2010: "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/ca2010/ca2010.mtx"
 * il2010: "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/il2010/il2010.mtx"
 * auto: "/afs/andrew.cmu.edu/usr21/preetang/private/courses/15618/15-618-project/data/auto/auto.mtx" <unweighted>
 * delaunay: "/afs/andrew.cmu.edu/usr21/preetang/private/courses/15618/15-618-project/data/delaunay_n24/delaunay_n24.mtx" <unweighted>
 * europe_osm: "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/europe_osm/europe_osm.mtx" <unweighted>
 * 333SP: "/afs/andrew.cmu.edu/usr21/preetang/private/courses/15618/15-618-project/data/333SP/333SP.mtx" <unweighted>
 * venturi: "/afs/andrew.cmu.edu/usr21/preetang/private/courses/15618/15-618-project/data/venturiLevel3/venturiLevel3.mtx" <unweighted>
 * rgg : "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/rgg_n_2_21_s0/rgg_n_2_21_s0.mtx"
 * kron19: "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/kron_g500-logn19/kron_g500-logn19.mtx"
 * kron20: "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/kron_g500-logn20/kron_g500-logn20.mtx"
 * kron20: "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/kron_g500-logn21/kron_g500-logn21.mtx"
 */

int main()
{
	Graph* g = new Graph();

	char* filename = (char *)"/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/gautamj+preetang/data/kron_g500-logn19/kron_g500-logn19.mtx";
	std::cout << "Reading " << filename << "\n";

	double startTime = CycleTimer::currentSeconds();
	g->ReadGraph(filename, true, false);
	double endTime = CycleTimer::currentSeconds();

	std::cout << "Graph Read. Found " << g->GetNodes() << " nodes and " << g->GetEdges() << " edges" << std::endl;
	std::cout << "Read completed in " << (endTime - startTime) << std::endl;

	int* neighbours = g->GetNeighbours();
	int* offsets = g->GetOffsets();
	int n = g->GetNodes();
	int m = g->GetEdges();
	int* visitedGlobal = (int *)calloc(g->GetNodes()+1, sizeof(int));

	int numBFSCalls = 0;
	double timeForSeqBFS = 0.0;
	int root = 1;
	while (root != 0) {
		// Init the visited array for this BFS
		int* visited = (int *)calloc(g->GetNodes()+1, sizeof(int));
		visited[root] = 1;

		// Start BFS with this root
		startTime = CycleTimer::currentSeconds();
		BFSSeq(g, root, visited);
		endTime = CycleTimer::currentSeconds();

		// move visited to visitedGlobal and get next root
		std::pair<int, int> update = updateGlobalVisited(visitedGlobal, visited, n);
		if (update.second >= UPDATE_THRESHOLD) {
			// Add to total time
			timeForSeqBFS += (endTime - startTime);
			numBFSCalls++;
			// break;
		}
		root = update.first;

		free(visited);
	}
	double avgSeqTime = (timeForSeqBFS / numBFSCalls);
	printf("Sequential Single Thread BFS Time: %.3f ms\n", 1000.f * avgSeqTime);
	// std::cout << "Sequential Single Thread BFS completed in " << (timeForSeqBFS / numBFSCalls)  * 1000.f << " ms" << std::endl;

	// check the state of the gpu
	printCudaInfo();

	// run the baseline cuda program (timing starts inside)
	int* levelsBase = (int *) calloc((n + 1), sizeof(int));
    BfsCuda(n, m, offsets, neighbours, levelsBase, 0, UPDATE_THRESHOLD);
    free(levelsBase);

    // run the baseline cuda program (timing starts inside)
	int* levels = (int *) calloc((n + 1), sizeof(int));
    BfsCuda(n, m, offsets, neighbours, levels, 1, UPDATE_THRESHOLD);

	// checking GPU output
	bool correct = true;
	for (int i = 1; i <= n; i++) {
		if (visitedGlobal[i] != levels[i]) {
			std::cout << "Sequential and Parallel conflict at " << i << std::endl;
			std::cout << "Sequential level  " << visitedGlobal[i] << std::endl;
			std::cout << "Parallel level " << levels[i] << std::endl;
			correct = false;
			break;
		}
	} 

	std::cout << "Correctness : " << correct << "\n";

	// Create a map to analyze output levels
	std::map<int, int> levelCounts; 
	for (int i=0; i<=n; i++) {
		if (levelCounts.count(visitedGlobal[i]) == 0) {
			levelCounts[visitedGlobal[i]] = 0;
		}

		levelCounts[visitedGlobal[i]]++;
	}

	int totl = -1;
	for (std::map<int,int>::iterator it=levelCounts.begin(); it!=levelCounts.end(); it++) {
		std::cout << "# of Nodes with level " << it->first << ": " << it->second << std::endl;
		totl += it->second;
	}
	std::cout << "Total nodes " << n << ", accounted " << totl << std::endl;
	std::cout << "Total BFSs considered " << numBFSCalls << std::endl;

	free(visitedGlobal);
	free(levels);
	g->FreeGraph();
	delete(g);

	return 0;
}
#include <vector>

#define MAXLINE 1024

class Graph {
	public :
	Graph() {}

	int* GetOffsets() {
		return offsets;
	}

	int* GetNeighbours() {
		return neighbours;
	}

	int GetNodes() {
		return nnode;
	}

	int GetEdges() {
		return nedges;
	}

	void ReadGraph(const char* fname);

	void FreeGraph();

	private :
	int *offsets; // size : nnode + 2
	int *neighbours; // size : nedges
	int nnode;
	int nedges;
};
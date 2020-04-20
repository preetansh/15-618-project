#include <vector>

#define MAXLINE 1024

class Graph {
	public :
	Graph() {}

	std::vector<int> GetOffsets() {
		return offsets;
	}

	std::vector<int> GetNeighbours() {
		return neighbours;
	}

	int GetNodes() {
		return nnode;
	}

	int GetEdges() {
		return nedges;
	}

	void ReadGraph(const char* fname);

	private :
	std::vector<int> offsets;
	std::vector<int> neighbours;
	int nnode;
	int nedges;
};
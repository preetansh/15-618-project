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

	int* GetParentOffsets() {
		return parent_offsets;
	}

	int* GetParents() {
		return parents;
	}

	bool* GetLeaves() {
		return leaves;
	}

	bool* GetRoots() {
		return roots;
	}

	int* GetChildToParentIndex() {
		return child_to_parent_index;
	}

	int* GetParentToChildIndex() {
		return parent_to_child_index;
	}

	void ReadGraph(const char* fname, bool isWeighted, bool keepDirection);

	int indexOfChild(int child, int parent);

	void ReadDFSGraph(const char* fname, bool isWeighted);

	void FreeGraph();

	private :
	int *offsets; // size : nnode + 2
	int *neighbours; // size : bfs : nedges, dfs : nedges + nroots
	int nnode;
	int nedges;

	// populate for DFS
	int* parent_offsets; // size : nnode + 2
	int *parents; // essentially reversing the neighbours, size : nedges
	bool *leaves; // leaves[i] = 1, i is a leaf, size : nnode + 1
	bool *roots; // roots[i] = 1, i is a root. Although the grand root will be zero, size : nnode + 1
	int* child_to_parent_index; // size : nedges + nroots, to store the position of a child in the adjacency list of parent  
	int* parent_to_child_index; // size : nedges + nroots, to store the position of a parent in the adjacency list of child
};
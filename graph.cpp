#include "graph.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <string.h>
#include <map>
// #include <tuple>

/* See whether line of text is a comment */
bool is_comment(char *s) {
    int i;
    int n = strlen(s);
    for (i = 0; i < n; i++) {
	char c = s[i];
	if (!isspace(c))
	    return c == '%';
    }
    return false;
}

void Graph::ReadGraph(const char* fname, bool isWeighted, bool keepDirection) {

	FILE *fp;
	char linebuf[MAXLINE];
    int lineno = 0;

    int row,col,nnz;

    // try to open the file
    if ((fp = fopen(fname, "r")) == NULL) 
 	{
      std::cerr << "Cannot open filename" << fname << std::endl;
      exit(1);
  	}

    // Read header information
    while (fgets(linebuf, MAXLINE, fp) != NULL) {
		lineno++;
		if (!is_comment(linebuf))
		    break;
	}

	if (sscanf(linebuf, "%d %d %d", &row, &col, &nnz) != 3) {
		std::cout << "ERROR. Malformed graph file header (line 1)\n";
		return;
    }

    // number of nodes is equal to row in the adjacency matrix
    nnode = row;
    if (keepDirection) {
        nedges = nnz; // Only one direction
    }
    else {
        nedges = nnz * 2; // Undirected edge, v1 -> v2, v2 -> v1
    }


    // initialization of the lists
    offsets = (int *) malloc(sizeof(int) * (nnode + 2)); // 0th node and nnode + 1 node
    // 0 for easy index and nnode + 1 for easy offsets

    neighbours = (int *) malloc(sizeof(int) * (nedges)); // all edges 

    std::map<int, std::vector<int> > adj_lists;


    int edge_from = -1, edge_to = -1;
    for (int i = 0; i < nnz; i++) {
        if (!isWeighted) {
            fscanf(fp, "%d %d\n", &edge_from, &edge_to);
        } else {
            int weight;
            fscanf(fp, "%d %d %d\n", &edge_from, &edge_to, &weight);
        }

        if (!(keepDirection)) {
            if (adj_lists.count(edge_from) == 0) {
                std::vector<int> from_list;
                from_list.push_back(edge_to);
                adj_lists[edge_from] = from_list;
            }
            else {
                adj_lists[edge_from].push_back(edge_to);
            }
        }

    	if (adj_lists.count(edge_to) == 0) {
    		std::vector<int> to_list;
    		to_list.push_back(edge_from);
    		adj_lists[edge_to] = to_list;
    	}
    	else {
    		adj_lists[edge_to].push_back(edge_from);
    	}

    }

    offsets[0] = 0;

    int prev_offset = 0;
    int neighbour_current = 0; // position at which adj_lists[i] needs to be inserted

    for (int i = 1; i <= nnode; i++) {
    	if (adj_lists.count(i) == 0) {
    		offsets[i] = prev_offset;
    	}
    	else {
    		offsets[i] = prev_offset;
    		prev_offset += adj_lists[i].size();
            std::copy(adj_lists[i].begin(), adj_lists[i].end(), &neighbours[neighbour_current]);
    		neighbour_current += adj_lists[i].size();
    	}
    }

    offsets[nnode + 1] = prev_offset;
}

int Graph::indexOfChild(int child, int parent) {
    int o = offsets[parent];
    int num_children = offsets[parent + 1] - offsets[parent];
    for (int i = 0; i < num_children; i++) {
        if (neighbours[o + i] == child) {
            return (o + i); 
        }
    }
    return -1;
}

// A function to read the graph for DFS
void Graph::ReadDFSGraph(const char* fname, bool isWeighted) {

    FILE *fp;
    char linebuf[MAXLINE];
    int lineno = 0;

    int row,col,nnz;
    std::map<std::pair<int,int>, int> edge_to_index;

    // try to open the file
    if ((fp = fopen(fname, "r")) == NULL) 
    {
      std::cerr << "Cannot open filename" << fname << std::endl;
      exit(1);
    }

    // Read header information
    while (fgets(linebuf, MAXLINE, fp) != NULL) {
        lineno++;
        if (!is_comment(linebuf))
            break;
    }

    if (sscanf(linebuf, "%d %d %d", &row, &col, &nnz) != 3) {
        std::cout << "ERROR. Malformed graph file header (line 1)\n";
        return;
    }

    // number of nodes is equal to row in the adjacency matrix
    nnode = row;
    nedges = nnz; // one direction edges, DAG


    // initialization of the lists
    offsets = (int *) malloc(sizeof(int) * (nnode + 2)); // 0th node and nnode + 1 node
    parent_offsets = (int *) malloc(sizeof(int) * (nnode + 2));
    // 0 for easy index and nnode + 1 for easy offsets

    roots = (bool *) malloc(sizeof(int) * (nnode + 1)); // 0th node included
    memset(roots, true, (nnode + 1) * sizeof(bool));
    leaves = (bool *) malloc(sizeof(int) * (nnode + 1)); // 0th node included
    memset(leaves, true, (nnode + 1) * sizeof(bool)); 
    leaves[0] = 0;

    std::map<int, std::vector<int> > adj_lists;
    std::map<int, std::vector<int> > parent_lists; // store the parents

    std::cout << "Starting to read" << "\n";

    int edge_from = -1, edge_to = -1;
    for (int i = 0; i < nnz; i++) {
        if (!isWeighted) {
            fscanf(fp, "%d %d\n", &edge_from, &edge_to);
        } else {
            float weight;
            fscanf(fp, "%d %d %f\n", &edge_from, &edge_to, &weight);
        }

        // set adjacency list
        if (adj_lists.count(edge_to) == 0) {
            std::vector<int> to_list;
            to_list.push_back(edge_from);
            adj_lists[edge_to] = to_list;
        }
        else {
            adj_lists[edge_to].push_back(edge_from);
        }

        // set parent list
        if (parent_lists.count(edge_from) == 0) {
            std::vector<int> from_list;
            from_list.push_back(edge_to);
            parent_lists[edge_from] = from_list;
        }
        else {
            parent_lists[edge_from].push_back(edge_to);
        }

        edge_to_index[std::pair<int,int> (edge_to, edge_from)] = i;
        roots[edge_from] = false;
        leaves[edge_to] = false;

    } // all edges done

    std::cout << "Edges read" << "\n";

    // find number of roots and put a back edge for 0
    std::vector<int> root_ids;

    for (int i = 1; i < (nnode + 1) ; i++) { // zero should not be counted here
        if (roots[i]) { 
            root_ids.push_back(i);
            std::vector<int> p_list;
            p_list.push_back(0);
            parent_lists[i] = p_list;
            edge_to_index[std::pair<int,int> (0, i)] = nnz + root_ids.size() - 1;
        }
    }

    int nroots = root_ids.size();

    neighbours = (int *) malloc(sizeof(int) * (nedges + nroots)); // all edges 
    parents = (int *) malloc(sizeof(int) * (nedges + nroots)); // all back - edges

    nedges += nroots; // insertion of new edges (0 -> roots)

    std::cout << "Roots identified" << "\n";
    offsets[0] = 0;
    parent_offsets[0] = 0;

    // make an edge from 0 to all roots, 0 becomes master root
    std::copy(root_ids.begin(), root_ids.end(), &neighbours[0]);

    int prev_offset = nroots;
    int neighbour_current = nroots; // position at which adj_lists[i] needs to be inserted

    int p_offset = 0;
    int p_current = 0;

    for (int i = 1; i <= nnode; i++) {
        // add children
        if (adj_lists.count(i) == 0) {
            offsets[i] = prev_offset;
        }
        else {
            offsets[i] = prev_offset;
            prev_offset += adj_lists[i].size();
            std::copy(adj_lists[i].begin(), adj_lists[i].end(), &neighbours[neighbour_current]);
            neighbour_current += adj_lists[i].size();
        }

        // add parents
        if (parent_lists.count(i) == 0) {
            parent_offsets[i] = p_offset;
        }
        else {
            parent_offsets[i] = p_offset;
            p_offset += parent_lists[i].size();
            std::copy(parent_lists[i].begin(), parent_lists[i].end(), &parents[p_current]);
            p_current += parent_lists[i].size();
        }
    }

    offsets[nnode + 1] = prev_offset;
    parent_offsets[nnode + 1] = p_offset;

    std::cout << "Adjacency list made" << "\n";

    // find the offsets of children in the parent adjacency list
    // parent_offsets, parents
    child_to_parent_index = (int *) malloc(sizeof(int) * (nedges));
    parent_to_child_index = (int *) malloc(sizeof(int) * (nedges));
    for (int i = 0; i <= nnode; i++) {
        int i_p_offset = parent_offsets[i];
        int num_parents = parent_offsets[i+1] - parent_offsets[i];
        for (int j = 0; j < num_parents; j++) {
            int i_p = parents[i_p_offset + j];
            int index_from_map = edge_to_index[std::pair<int,int> (i_p, i)];
            if (i_p == 0) {
                index_from_map -= nnz;
            }
            else {
                index_from_map += nroots;
            }
            child_to_parent_index[i_p_offset + j] = index_from_map;
            parent_to_child_index[index_from_map] = i_p_offset + j; 
        }
    }

    std::cout << "Child to parent index created" << "\n";
}


void Graph::FreeGraph() {
    free(offsets);
    free(neighbours);
    free(parents);
    free(roots);
    free(leaves);
    free(child_to_parent_index);
    free(parent_to_child_index);
} 
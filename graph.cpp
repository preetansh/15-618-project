#include "graph.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <string.h>
#include <map>

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

void Graph::ReadGraph(const char* fname) {

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
    nedges = nnz;

    std::map<int, std::vector<int> > adj_lists;


    int edge_from = -1, edge_to = -1;
    for (int i = 0; i < nnz; i++) {
    	fscanf(fp, "%d %d\n", &edge_from, &edge_to);

    	if (adj_lists.count(edge_from) == 0) {
    		std::vector<int> from_list;
    		from_list.push_back(edge_to);
    		adj_lists[edge_from] = from_list;
    	}
    	else {
    		adj_lists[edge_from].push_back(edge_to);
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

    offsets.push_back(0);

    int prev_offset = 0;
    for (int i = 1; i <= nnode; i++) {
    	if (adj_lists.count(i) == 0) {
    		offsets.push_back(prev_offset);
    	}
    	else {
    		offsets.push_back(prev_offset);
    		prev_offset += adj_lists[i].size();
    		neighbours.insert(neighbours.end(), adj_lists[i].begin(), adj_lists[i].end());
    	}
    	if (i % 10000 == 0) {
    		std::cout << i << std::endl;
    	}
    }

    offsets.push_back(prev_offset);
}
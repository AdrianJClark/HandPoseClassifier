#ifndef RANDOMTREE_H
#define RANDOMTREE_H

#include <istream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include "Features.h"

class RandomTree {
public:
	RandomTree() { left=0; right=0; EntropyScore=0; _class=-1;}
	~RandomTree() {}
	
	Feature splitFeature;
	RandomTree *left, *right;
	double EntropyScore;
	int _class;
};

void printTree(RandomTree *root, const char *filename) {
	std::vector<RandomTree*> vTree; vTree.push_back(root);
	for (int i=0; i<vTree.size(); i++) {
		if (vTree.at(i)->left!=0) vTree.push_back(vTree.at(i)->left);
		if (vTree.at(i)->right!=0) vTree.push_back(vTree.at(i)->right);
	}

	std::map<RandomTree*, int> mTree; mTree[0] = 0;
	for (int i=0; i<vTree.size(); i++) {
		if (vTree.at(i)!=0) mTree[vTree.at(i)] = i+1;
	}

	FILE *f = fopen(filename, "wb");
	fprintf(f, "Format: index, leftIndex, rightIndex, class - uX, uY, vX, vY, thresh - Score\r\n");
	for (std::vector<RandomTree*>::iterator i=vTree.begin(); i!=vTree.end(); i++) {	
		fprintf(f, "%d, %d, %d, %d - %d, %d, %d, %d, %lf - %f\r\n", 
			mTree[(*i)], mTree[(*i)->left], mTree[(*i)->right], (*i)->_class,
			(*i)->splitFeature.uX, (*i)->splitFeature.uY, 
			(*i)->splitFeature.vX, (*i)->splitFeature.vY, 
			(*i)->splitFeature.threshold, (*i)->EntropyScore);
	}
	fclose(f);
}

RandomTree *loadTree(const char* filename) {
	std::ifstream file(filename);
	std::string line;

	getline(file, line);

	int index, leftIndex, rightIndex, _class;
	int uX, uY, vX, vY;
	double thresh;

	struct TreeConnection {	int left, right; };
	std::map<int, TreeConnection> connections;
	std::map<int, RandomTree*> nodes;
	
	//Loop through the file and create nodes and record connections
	getline(file, line);
	while (sscanf(line.c_str(), "%d, %d, %d, %d - %d, %d, %d, %d, %lf", 
		&index, &leftIndex, &rightIndex, &_class, &uX, &uY, &vX, &vY, &thresh) == 9) {
		
		//Create Node
		RandomTree *node = new RandomTree();
		node->_class = _class;
		node->splitFeature.uX = uX; node->splitFeature.uY = uY; 
		node->splitFeature.vX = vX; node->splitFeature.vY = vY; 
		node->splitFeature.threshold = thresh;
		nodes[index] = node;

		//Create Connections
		TreeConnection tc; tc.left = leftIndex; tc.right = rightIndex;
		connections[index] = tc;

		getline(file, line);
	}

	//Connect nodes together into a tree
	for (std::map<int, RandomTree*>::iterator i=nodes.begin(); i!=nodes.end(); i++) {
		TreeConnection tc = connections[i->first];
		if (tc.left!=0) i->second->left = nodes[tc.left];
		if (tc.right!=0) i->second->right = nodes[tc.right];
	}

	//Return the root node
	return nodes[1];
}

#endif
#ifndef RANDOMTREE_H
#define RANDOMTREE_H

#include "Features.h"

class RandomTree {
public:
	RandomTree(int _id) { id = _id; left=0; right=0; EntropyScore=0; _class=-1;}
	~RandomTree() {}
	
	Feature splitFeature;
	RandomTree *left, *right;
	double EntropyScore;
	int _class;
	int id;
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

#endif
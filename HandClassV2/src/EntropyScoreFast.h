#ifndef ENTROPY_SCORE_FAST_H
#define ENTROPY_SCORE_FAST_H

#include <vector>
#include <map>
#include <cmath>

using namespace std;

void getClassCount(vector<int> classes, map<int, int>& classCounts, int numDiffClasses);
void printClassCounts(map<int, int> classCounts, int numDiffClasses);
double calculateGain(vector<int> classes, vector<int> cLeft, vector<int> cRight, int numDiffClasses);
inline double log2( double n );
int calculateCardinality(vector<int> classes, std::map<int, int> classCounts, int numDiffClasses);
double calculateEntropy(vector<int> classes, std::map<int, int> cCounts, int numDiffClasses);

void getClassCount(vector<int> classes, map<int, int>& classCounts, int numDiffClasses) {
	for (int i=0; i<=numDiffClasses; i++) classCounts[i]=0;
	for (unsigned int i=0; i<classes.size(); i++) classCounts[classes.at(i)]++;
}

void printClassCounts(map<int, int> classCounts, int numDiffClasses) {
	for (int i=0; i<=numDiffClasses; i++) printf("%2d ", i); printf("\n");
	for (int i=0; i<=numDiffClasses; i++) printf("%2d ", classCounts[i]); printf("\n");

}

double calculateGain(vector<int> classes, vector<int> cLeft, vector<int> cRight, int numDiffClasses) {
	std::map<int, int> cCounts; getClassCount(classes, cCounts, numDiffClasses);
	std::map<int, int> cLCounts; getClassCount(cLeft, cLCounts, numDiffClasses);
	std::map<int, int> cRCounts; getClassCount(cRight, cRCounts, numDiffClasses);

	double HC = classes.size()==0?0:calculateEntropy(classes, cCounts, numDiffClasses);
	double HLC = cLeft.size()==0?0:calculateEntropy(cLeft, cLCounts, numDiffClasses);
	double HRC = cRight.size()==0?0:calculateEntropy(cRight, cRCounts, numDiffClasses);

	double LC = double(classes.size());
	double LLC = double(cLeft.size());
	double LRC = double(cRight.size());

	double gain = HC - (((LLC/LC)*HLC)+((LRC/LC)*HRC));

	return gain;
}

inline double log2( double n ) {  
	return n==0?0:log(n)/log(2.);  
}

double calculateEntropy(vector<int> classes, std::map<int, int> classCounts, int numDiffClasses) {

	std::map<int, double> classCountsNormalized;
	for (int i=0; i<=numDiffClasses; i++)
		classCountsNormalized[i] = (double)(classCounts[i])/(double)(classes.size());

	double entropy = 0;
	for (int i=0; i<=numDiffClasses; i++) {
		entropy += -(classCountsNormalized[i] * log2(classCountsNormalized[i]));
	}
	return entropy;

}

#endif
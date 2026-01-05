#include <windows.h>
#include "EntropyScore.h"

void generateData(vector<int>& classes, int numClasses, int classCount);
void randomSplitData(vector<int> classes, vector<int>& cLeft, vector<int>& cRight);
void deliberateSplitData(vector<int> classes, vector<int>& cLeft, vector<int>& cRight, int numDiffClasses);

void main() {

	//Containers for class data
	vector<int> classes; map<int, int> classesCount;

	//Generate Random Class Data
	generateData(classes, 20, 10000);

	LARGE_INTEGER freq, start, end;
	QueryPerformanceFrequency(&freq);

	//Random Score
	{
		//Generate Random Split
		vector<int> cLeft, cRight; map<int, int> clCount, crCount;
		randomSplitData(classes, cLeft, cRight);

		//Calculate Score for Split
		QueryPerformanceCounter(&start);
		double score = calculateGain(classes, cLeft, cRight, 20);
		QueryPerformanceCounter(&end);
		printf("%f\n", score);
		printf("time: %f\n", double(end.QuadPart-start.QuadPart)/double(freq.QuadPart));
	}

	//Deliberate Score
	{
		//Generate Perfect Split
		vector<int> cLeft, cRight; map<int, int> clCount, crCount;
		deliberateSplitData(classes, cLeft, cRight, 20);

		//Calculate Score for Split
		printf("%f\n", calculateGain(classes, cLeft, cRight, 20));
	}

}


void generateData(vector<int>& classes, int numDiffClasses, int classCount) {
	for (int i=0; i<classCount; i++) {
		int classLabel = int((double(rand())/double(RAND_MAX))*(numDiffClasses))+1;
		classes.push_back(classLabel);
	}
}


void randomSplitData(vector<int> classes, vector<int>& cLeft, vector<int>& cRight) {
	for (unsigned int i=0; i<classes.size(); i++) {
		int direction = int((double(rand())/double(RAND_MAX))*2);
		if (classes.at(i)%2==1) {
			cLeft.push_back(classes.at(i));
		} else {
			if (direction==0) cLeft.push_back(classes.at(i));
			else cRight.push_back(classes.at(i));
		}
	}
}

void deliberateSplitData(vector<int> classes, vector<int>& cLeft, vector<int>& cRight, int numDiffClasses) {
	for (unsigned int i=0; i<classes.size(); i++) {
		if (classes.at(i)%2==1) {
			cLeft.push_back(classes.at(i));
		} else {
			cRight.push_back(classes.at(i));
		}
	}
}


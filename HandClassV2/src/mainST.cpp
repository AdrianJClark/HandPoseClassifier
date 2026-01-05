#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <windows.h>

#include "Settings.h"
#include "io.h"
#include "features.h"
#include "EntropyScoreFast.h"
#include "RandomTree.h"

struct ImagePoint {
	int frame, x, y, _class;
};


void calculateImagePoints(vector<ImagePoint> &ImagePoints);
void split(RandomTree *root, vector<ImagePoint> imagePoints, vector<Feature> features, int depth);
double GetFeatureValue(IplImage* depthIm, int x, int y, Feature feature);

std::map<int, IplImage*> ImageBuffer;

LARGE_INTEGER freq;

void main() {

	QueryPerformanceFrequency(&freq);

	srand ( (unsigned int) time(NULL) );

	loadSettings("Settings.xml");

	vector<Feature> features = loadUV(featuresFileName.c_str());

	vector<ImagePoint> imagePoints;
	calculateImagePoints(imagePoints);

	//Buffer all the images we need for speed
	for (unsigned int i=0; i<imagePoints.size(); i++) {
		if (ImageBuffer.find(imagePoints.at(i).frame)==ImageBuffer.end()) {
			ImageBuffer[imagePoints.at(i).frame] = getDepthImage(imagePoints.at(i).frame, depthPath.c_str());
		}
	}

	LARGE_INTEGER start, end;
	QueryPerformanceCounter(&start);
	RandomTree *root = new RandomTree(1); int depth = 0;
	split(root, imagePoints, features, depth);
	QueryPerformanceCounter(&end);

	//Empty Image Buffer
	for (std::map<int, IplImage*>::iterator i = ImageBuffer.begin(); i!=ImageBuffer.end(); i++) {
		cvReleaseImage(&(i->second));
	}

	printf("Elapsed Time: %fs\r\n", double(end.QuadPart-start.QuadPart)/double(freq.QuadPart));
	printTree(root, treeFileName.c_str());
}

bool containsSingleClass(vector<ImagePoint> ip) {
	if (ip.size()<2) return true;
	int fClass = ip.at(0)._class;
	for (int i=1; i<ip.size(); i++) {
		if (ip.at(i)._class != fClass) {
			return false;
		}
	}
	return true;
}

int getMaximumClass(vector<ImagePoint> imagePoints, int numDiffClasses) {
	std::map<int, int> classCounts; 
	//Initialize class counts to 0
	for (int i=0; i<=numDiffClasses; i++) classCounts[i]=0;
	
	//Increase class Counts size
	for (int i=0; i<imagePoints.size(); i++) classCounts[imagePoints.at(i)._class]++;
	
	//Find maximum count number
	int maxCount=0; int maxIndex=0;
	for (int i=0; i<=numDiffClasses; i++) {
		if (classCounts[i]>maxCount) { maxCount=classCounts[i]; maxIndex = i; }
	}

	return maxIndex;
}

void split(RandomTree *node, vector<ImagePoint> imagePoints, vector<Feature> features, int depth) {

	//If we've only got one class left, set that and return
	if (containsSingleClass(imagePoints)) {
		node->_class = imagePoints.at(0)._class;
		node->splitFeature.uX = node->splitFeature.vX = node->splitFeature.uY = node->splitFeature.vY = node->splitFeature.threshold = 0;
		return;
	}

	//If we've exceeded the depth, set the class to the mode and quit.
	if (depth>maxTreeDepth) {
		node->_class = getMaximumClass(imagePoints, classCount);
		return;
	}

	LARGE_INTEGER tStart, tEnd;
	vector<double> featureEntropy;

	for (unsigned int i=0; i<features.size(); i++) {
		vector<double> featureValues;

		QueryPerformanceCounter(&tStart);
		//Calculate the values for each image point
		for (unsigned int j=0; j<imagePoints.size(); j++) {
			//Process the image point
			featureValues.push_back(GetFeatureValue(ImageBuffer[imagePoints.at(j).frame], imagePoints.at(j).x, imagePoints.at(j).y, features.at(i)));
		}

		vector<int> beforeSplit, leftSplit, rightSplit;

		//Loop through all the features, add each class to before split
		//And split the classes based on the threshold value
		for (unsigned int j=0; j<featureValues.size(); j++) {
			beforeSplit.push_back(imagePoints.at(j)._class);
			if (featureValues.at(j)<features.at(i).threshold) {
				leftSplit.push_back(imagePoints.at(j)._class);
			} else {
				rightSplit.push_back(imagePoints.at(j)._class);
			}
		}

		//Calculate the entropy for this feature and add it to the list
		featureEntropy.push_back(calculateGain(beforeSplit, leftSplit, rightSplit, classCount));

		QueryPerformanceCounter(&tEnd);

		printf("Processing: %d/%d (%d%%). Time Remaining: %ds\r", 
			i, features.size(), int(double(i)/double(features.size())*100.0), 
			int((double(tEnd.QuadPart-tStart.QuadPart)/double(freq.QuadPart))*(features.size()-i)));
		
	}

	//Find the Feature with the Maximum Entropy 
	double maxEntropyVal = -DBL_MAX; int maxEntropyIndex = 0;
	for (int i=0; i<featureEntropy.size(); i++) {
		if (featureEntropy.at(i) > maxEntropyVal) {
			maxEntropyVal = featureEntropy.at(i); maxEntropyIndex = i;
		}
	}

	//Store the Entropy Score and Feature
	node->EntropyScore = maxEntropyVal;
	node->splitFeature = features.at(maxEntropyIndex);

	//Create a new feature set with the current one removed so the children can't use it
	std::vector<Feature> featuresNew; 
	featuresNew.insert(featuresNew.begin(), features.begin(), features.end());
	featuresNew.erase(featuresNew.begin() + maxEntropyIndex);

	//Split the image points into left and right branches
	vector<ImagePoint> imagePointsLeft, imagePointsRight;
	for (unsigned int j=0; j<imagePoints.size(); j++) {
		double val = GetFeatureValue(ImageBuffer[imagePoints.at(j).frame], imagePoints.at(j).x, imagePoints.at(j).y, features.at(maxEntropyIndex));
		if (val<features.at(maxEntropyIndex).threshold) {
			imagePointsLeft.push_back(imagePoints.at(j));
		} else {
			imagePointsRight.push_back(imagePoints.at(j));
		}
	}

	exit(0);
	if (imagePointsLeft.size()>0) {
		node->left = new RandomTree(node->id*2);
		split(node->left, imagePointsLeft, featuresNew, depth+1);
	}

	if (imagePointsRight.size()>0) {
		node->right = new RandomTree(node->id*2+1);
		split(node->right, imagePointsRight, featuresNew, depth+1);
	}

	return ;
}

void calculateImagePoints(vector<ImagePoint> &ImagePoints) {
	//Calculate the image points for each image
	for (int frame=frameBegin; frame<=frameEnd; frame++) {
		IplImage *classIm = getClassImage(frame, classPath.c_str());
		IplImage *depthIm = getDepthImage(frame, depthPath.c_str());

		//Get a mask for the classified image
		IplImage *classImMask = cvCloneImage(classIm);
		int nonZero = cvCountNonZero(classImMask);

		//Calculate the maximum number of points to find
		int pMax = (classIm->width*classIm->height)/10;
		if (pMax>float(nonZero)*.9) pMax = float(nonZero)*.9;

		//Loop until we have enough points
		int pCount=0;
		while (pCount<pMax) {
			//Grab a point on the image
			int x = (double(rand())/double(RAND_MAX))*double(classIm->width-1);
			int y = (double(rand())/double(RAND_MAX))*double(classIm->height-1);

			//If it's a valid point
			if (CV_IMAGE_ELEM(classImMask, unsigned char, y, x)!=0) {
				CV_IMAGE_ELEM(classImMask, unsigned char, y, x)=0;

				//Create the image point and push it on the stack
				ImagePoint ip; 
				ip.frame = frame; ip.x = x; ip.y = y; 
				ip._class = CV_IMAGE_ELEM(classIm, unsigned char, y, x);
				ImagePoints.push_back(ip);

				//Increment the counter
				pCount++;
			}
		}

		//Release the Mask
		cvReleaseImage(&classImMask);

		cvReleaseImage(&classIm);
		cvReleaseImage(&depthIm);
	}
}



double GetFeatureValue(IplImage* depthIm, int x, int y, Feature feature) {
	//Calculate the vector position
	float dix = 1.0/(CV_IMAGE_ELEM(depthIm, float, y, x)*0.005);
	float uxd = feature.uX*dix, uyd = feature.uY*dix;
	float vxd = feature.vX*dix, vyd = feature.vY*dix;
	float uX = x + uxd, uY = y + uyd;
	float vX = x + vxd, vY = y + vyd;

	//Check that the positions are within bounds
	float diU, diV;
	if (uX<0 || uY<0 || uX>=depthIm->width || uY>=depthIm->height) {
		diU = 10000;
	} else {
		diU = CV_IMAGE_ELEM(depthIm, float, (int)uY, (int)uX);
	}
	if (vX<0 || vY<0 || vX>=depthIm->width || vY>=depthIm->height) {
		diV = 10000;
	} else {
		diV = CV_IMAGE_ELEM(depthIm, float, (int)vY, (int)vX);
	}

	//If they fall on the background, set them to a high number
	if (diU==0.0) diU = 10000;
	if (diV==0.0) diV = 10000;

	//Return the value
	return diU-diV;
}
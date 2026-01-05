// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <windows.h>

#include "Settings.h"
#include "io.h"
#include "Features.h"
#include "EntropyScoreFast.h"
#include "RandomTree.h"

struct ImagePoint {
	int frame, x, y, _class;
};


void calculateImagePoints(vector<ImagePoint> &ImagePoints);
void split(RandomTree *root, vector<ImagePoint> imagePoints, vector<Feature> features, int depth);
double GetFeatureValue(IplImage* depthIm, int x, int y, Feature feature);

//Cuda Functions
extern "C" void cudaInit(const int argc, const char** argv);
extern "C" void cudaLoadImageData(float *vImageData, int imWidth, int imHeight, int imCount);
extern "C" void cudaFreeImageData();
extern "C" void cudaStoreImagePoints(int2 *ImagePointsPos, int2 *ImagePointsData, int ImagePointCount);
extern "C" void cudaFreeImagePoints();
extern "C" void cudaGetFeatureValues(double *featureValues, int2 featureU, int2 featureV);
extern "C" void cudaSplitImagePoints(bool *splitLeft, int2 featureU, int2 featureV, double featureT);
extern "C" void cudaCalculateFeatureGain(int2* featureU, int2* featureV, double *featureT, int featureCount, int classCount, double *FeatureGain);

LARGE_INTEGER freq;

void main() {

	QueryPerformanceFrequency(&freq);

	srand ( (unsigned int) time(NULL) );

	loadSettings("Settings.xml");

	vector<Feature> features = loadUV(featuresFileName.c_str());

	vector<ImagePoint> imagePoints;
	calculateImagePoints(imagePoints);

	//Buffer all the images in the GPU
	{
		//Find the size of the images
		IplImage *tmpImage = getDepthImage(0, depthPath.c_str());
		int imWidth = tmpImage->width, imHeight = tmpImage->height;
		int imCount = (frameEnd-frameBegin)+1;
		cvReleaseImage(&tmpImage);

		//Allocate the data
		float *allImageData = (float*)malloc(imWidth*imHeight*imCount*sizeof(float));
		for (unsigned int i=frameBegin; i<=frameEnd; i++) {
			float *framePtr = allImageData+i*(imWidth*imHeight);
			IplImage *tmpImage = getDepthImage(i, depthPath.c_str());
			memcpy(framePtr, tmpImage->imageData, imWidth*imHeight*sizeof(float));
			cvReleaseImage(&tmpImage);
		}

		//Copy data into cuda
		cudaLoadImageData(allImageData, imWidth, imHeight, imCount);

		//Release data
		free(allImageData);
	}

	LARGE_INTEGER start, end;
	QueryPerformanceCounter(&start);
	RandomTree *root = new RandomTree(1); int depth = 0;
	split(root, imagePoints, features, depth);
	QueryPerformanceCounter(&end);

	//Clean up Cuda Data
	cudaFreeImageData();
    cudaDeviceReset();


	printf("Elapsed Time: %fs                \r\n", double(end.QuadPart-start.QuadPart)/double(freq.QuadPart));
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
	QueryPerformanceCounter(&tStart);
	
	//Copy all the Image Points to CUDA device memory
	{
		//Allocate Memory	
		int2 *IPPosition = (int2*)malloc(sizeof(int2)*imagePoints.size());
		int2 *IPData = (int2*)malloc(sizeof(int2)*imagePoints.size());
		//Copy in the Data
		for (unsigned int j=0; j<imagePoints.size(); j++) {
			IPPosition[j] = make_int2(imagePoints.at(j).x, imagePoints.at(j).y);
			IPData[j] = make_int2(imagePoints.at(j).frame, imagePoints.at(j)._class);
		}
		//Sore it in cuda
		cudaStoreImagePoints(IPPosition, IPData, imagePoints.size());
		free(IPPosition); free(IPData);
	}

	//Calculate approximately how many features we can process at once
	int blockSize = 500000000/ (imagePoints.size() * sizeof(bool)); 
	if (blockSize>features.size()) blockSize = features.size();

	//Copy all of the Feature information into cuda
	int2 *featureU = (int2*)malloc(sizeof(int2)*blockSize);
	int2 *featureV = (int2*)malloc(sizeof(int2)*blockSize);
	double *featureT = (double*)malloc(sizeof(double)*blockSize);
	for (unsigned int j=0; j<blockSize; j++) {
		featureU[j] = make_int2(features.at(j).uX, features.at(j).uY);
		featureV[j] = make_int2(features.at(j).vX, features.at(j).vY);
		featureT[j] = features.at(j).threshold;
	}

	//Create space for the information gain
	double* featureGain = (double*)malloc(blockSize * sizeof(double));
	//Calculate information gain using CUDA
	cudaCalculateFeatureGain(featureU, featureV, featureT, blockSize, classCount, featureGain);
	//Clean up some memory
	free(featureU); free(featureV); free(featureT);

	//Find the Feature with the Maximum Entropy 
	double maxEntropyVal = -DBL_MAX; int maxEntropyIndex = 0;
	for (int i=0; i<blockSize; i++) {
		if (featureGain[i] > maxEntropyVal) {
			maxEntropyVal = featureGain[i]; maxEntropyIndex = i;
		}
	}

	//Store the Entropy Score and Feature
	node->EntropyScore = maxEntropyVal;
	node->splitFeature = features.at(maxEntropyIndex);

	//Use Cuda to calculate the split
	vector<ImagePoint> imagePointsLeft, imagePointsRight;
	bool *splitLeft = (bool*)malloc(sizeof(bool)*imagePoints.size());
	int2 fU = make_int2(features.at(maxEntropyIndex).uX,features.at(maxEntropyIndex).uY);
	int2 fV = make_int2(features.at(maxEntropyIndex).vX, features.at(maxEntropyIndex).vY);
	cudaSplitImagePoints(splitLeft, fU, fV, features.at(maxEntropyIndex).threshold);

	//Split the image points into left and right branches
	for (unsigned int j=0; j<imagePoints.size(); j++) {
		if (splitLeft[j]) {
			imagePointsLeft.push_back(imagePoints.at(j));
		} else {
			imagePointsRight.push_back(imagePoints.at(j));
		}
	}
	free(splitLeft);

	//Clean up the image points
	cudaFreeImagePoints();

	QueryPerformanceCounter(&tEnd);


	printf("Time taken to split: %f\r", double(tEnd.QuadPart-tStart.QuadPart)/double(freq.QuadPart));

	//Split into subtrees
	if (imagePointsLeft.size()>0) {
		node->left = new RandomTree(node->id*2);
		split(node->left, imagePointsLeft, features, depth+1);
	}

	if (imagePointsRight.size()>0) {
		node->right = new RandomTree(node->id*2+1);
		split(node->right, imagePointsRight, features, depth+1);
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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <windows.h>

#include "Features.h"
#include "RandomTree.h"
#include "ColorConvert.h"

struct ImagePoint {
	int frame, x, y, _class;
};

using namespace std;

int frameBegin=0; int frameEnd=95;

void calculateImagePoints(IplImage *depthIm, vector<ImagePoint> &ImagePoints);
void getAllImagePoints(IplImage *depthIm, vector<ImagePoint> &ImagePoints);
void predict(RandomTree *root, IplImage *depthIm, vector<ImagePoint> &ImagePoints);
double GetFeatureValue(IplImage* depthIm, int x, int y, Feature feature);
IplImage* getDepthMask(IplImage *depth);

void main() {

	srand ( (unsigned int) time(NULL) );

	loadColorTable("Data/ColorTable.txt");
	RandomTree *root = loadTree("Data/RotationTreeCUDA.txt");

	for (int i=frameBegin; i<=frameEnd; i++) {
		char filename[50]; sprintf(filename, "Data/Rotation/depth%d.yml", i);
		CvFileStorage* fs = cvOpenFileStorage( filename, 0, CV_STORAGE_READ );
		IplImage *depthIm = (IplImage*)cvRead(fs, cvGetFileNodeByName( fs, NULL, "depth" ));
		cvReleaseFileStorage(&fs);

		vector<ImagePoint> imagePoints;
		getAllImagePoints(depthIm, imagePoints);
	
		predict(root, depthIm, imagePoints);

		IplImage *classImage = cvCreateImage(cvGetSize(depthIm), IPL_DEPTH_8U, 3); cvSetZero(classImage);
		for (int i=0; i<imagePoints.size(); i++) {
			CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3) = ColorMap[imagePoints.at(i)._class].b; 
			CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3+1) = ColorMap[imagePoints.at(i)._class].g;
			CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3+2) = ColorMap[imagePoints.at(i)._class].r;
		}

		cvShowImage("Classified", classImage);
		cvReleaseImage(&classImage);
		cvReleaseImage(&depthIm);
		switch (cvWaitKey(1)) {
			case 27: exit(0); break;
		}
	}

}

void getAllImagePoints(IplImage *depthIm, vector<ImagePoint> &ImagePoints) {
	//Get a mask for the classified image
	IplImage *depthImMask = getDepthMask(depthIm);

	for (int y=0; y<depthImMask->height; y++) {
		for (int x=0; x<depthImMask->width; x++) {
			if (CV_IMAGE_ELEM(depthImMask, unsigned char, y, x)!=0) {
			//Create the image point and push it on the stack
			ImagePoint ip; 
			ip.frame = 0; ip.x = x; ip.y = y; 
			ImagePoints.push_back(ip);
			}	
		}
	}

	//Release the Mask
	cvReleaseImage(&depthImMask);
}


void calculateImagePoints(IplImage *depthIm, vector<ImagePoint> &ImagePoints) {
	//Get a mask for the classified image
	IplImage *depthImMask = getDepthMask(depthIm);
	int nonZero = cvCountNonZero(depthImMask);

	//Calculate the maximum number of points to find
	int pMax = (depthImMask->width*depthImMask->height)/10;
	if (pMax>float(nonZero)*.9) pMax = float(nonZero)*.9;

	//Loop until we have enough points
	int pCount=0;
	while (pCount<pMax) {
		//Grab a point on the image
		int x = (double(rand())/double(RAND_MAX))*double(depthImMask->width-1);
		int y = (double(rand())/double(RAND_MAX))*double(depthImMask->height-1);

		//If it's a valid point
		if (CV_IMAGE_ELEM(depthImMask, unsigned char, y, x)!=0) {
			CV_IMAGE_ELEM(depthImMask, unsigned char, y, x)=0;

			//Create the image point and push it on the stack
			ImagePoint ip; 
			ip.frame = 0; ip.x = x; ip.y = y; 
			ImagePoints.push_back(ip);

			//Increment the counter
			pCount++;
		}
	}

	//Release the Mask
	cvReleaseImage(&depthImMask);
}

void predict(RandomTree *root, IplImage *depthIm, vector<ImagePoint> &imagePoints) {

	for (unsigned int i=0; i<imagePoints.size(); i++) {

		RandomTree *curNode = root;

		while (curNode->left!=0 || curNode->right!=0) {
			double value = GetFeatureValue(depthIm, imagePoints.at(i).x, imagePoints.at(i).y, curNode->splitFeature);
			if (value<curNode->splitFeature.threshold) {
				curNode=curNode->left;
			} else {
				curNode=curNode->right;
			}
		}
		imagePoints.at(i)._class = curNode->_class;
	}

	return;
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

IplImage* getDepthMask(IplImage *depth) {
	IplImage *depthMask = cvCreateImage(cvGetSize(depth), IPL_DEPTH_8U, 1); cvZero(depthMask);

	for (int y=0; y<depth->height; y++) {
		for (int x=0; x<depth->width; x++) {
			if (CV_IMAGE_ELEM(depth, float, y, x)!=0) 
				CV_IMAGE_ELEM(depthMask, unsigned char, y, x) = 255;
		}
	}

	return depthMask;
}

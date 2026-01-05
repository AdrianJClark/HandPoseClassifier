#if 1

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "src/Features.h"
#include "src/ColorConvert.h"
#include "src/RandomTree.h"

struct ImagePoint {
	int frame, x, y, _class;
};

void showScaledDepth(const char* windowName, IplImage *depthImage);
std::vector<Feature> features; int featIndex=0;
void predict(RandomTree *root, IplImage *depthIm, std::vector<ImagePoint> &imagePoints, float scale);
void getAllImagePoints(IplImage *depthIm, std::vector<ImagePoint> &ImagePoints);
double GetFeatureValue(IplImage* depthIm, int x, int y, Feature feature, float scale);

int cX=-1, cY=-1, tX=-1, tY=-1;

void CroppedDepthImageMouseCallback(int _event, int x, int y, int flags, void* param) {
	IplImage *croppedIm = (IplImage*)param;
	printf("x: %d, y: %d, z: %f\r", x, y, CV_IMAGE_ELEM(croppedIm, float, y, x));
	if (_event == CV_EVENT_LBUTTONDOWN) {
		cX = x; cY=y;
	} 
}

void TrainedDepthImageMouseCallback(int _event, int x, int y, int flags, void* param) {
	IplImage *croppedIm = (IplImage*)param;
	printf("x: %d, y: %d, z: %f\r", x, y, CV_IMAGE_ELEM(croppedIm, float, y, x));
	if (_event == CV_EVENT_LBUTTONDOWN) {
		tX=x; tY=y;
	} 
}


void main() {

	features =  loadUV("Data/features.txt");
	loadColorTable("Data/ColorTable.txt");
	RandomTree *root = loadTree("Data/TranslationTree.txt");

	CvFileStorage* fstorage = cvOpenFileStorage("croppedDepth.yml", NULL, CV_STORAGE_READ);
    IplImage *storedDepthImage = (IplImage*)cvRead( fstorage, cvGetFileNodeByName( fstorage, NULL, "depth" ));
	cvReleaseFileStorage(&fstorage);

	fstorage = cvOpenFileStorage("Data/Translation/depth0.yml", NULL, CV_STORAGE_READ);
    IplImage *trainedDepthImage = (IplImage*)cvRead( fstorage, cvGetFileNodeByName( fstorage, NULL, "depth" ));
	cvReleaseFileStorage(&fstorage);

	cvNamedWindow("Cropped"); cvNamedWindow("Training");
	cvSetMouseCallback("Cropped", CroppedDepthImageMouseCallback, storedDepthImage);
	cvSetMouseCallback("Training", TrainedDepthImageMouseCallback, trainedDepthImage);
	
	bool running = true; float scale = 0.04;
	while (running) {
		if (cX !=-1 && cY!=-1) {
			Feature feature = features.at(featIndex);
			float dix = 1.0/(CV_IMAGE_ELEM(storedDepthImage, float, cY, cX)*scale);
			float uxd = feature.uX*dix, uyd = feature.uY*dix;
			float vxd = feature.vX*dix, vyd = feature.vY*dix;
			float uX = cX + uxd, uY = cY + uyd;
			float vX = cX + vxd, vY = cY + vyd;
			IplImage *outImage = cvCreateImage(cvGetSize(storedDepthImage), IPL_DEPTH_8U, 3);
			cvConvertImage(storedDepthImage, outImage);
			cvLine(outImage, cvPoint(cX, cY), cvPoint(uX, uY), cvScalar(255,128,128));
			cvLine(outImage, cvPoint(cX, cY), cvPoint(vX, vY), cvScalar(128,255,128));
			cvShowImage("Cropped", outImage);
			cvReleaseImage(&outImage);
		} else {
			cvShowImage("Cropped", storedDepthImage);
		}

		if (tX !=-1 && tY!=-1) {
			Feature feature = features.at(featIndex);
			float dix = 1.0/(CV_IMAGE_ELEM(trainedDepthImage, float, tY, tX)*0.005);
			float uxd = feature.uX*dix, uyd = feature.uY*dix;
			float vxd = feature.vX*dix, vyd = feature.vY*dix;
			float uX = tX + uxd, uY = tY + uyd;
			float vX = tX + vxd, vY = tY + vyd;
			IplImage *outImage = cvCreateImage(cvGetSize(trainedDepthImage), IPL_DEPTH_8U, 3);
			cvConvertImage(trainedDepthImage, outImage);
			cvLine(outImage, cvPoint(tX, tY), cvPoint(uX, uY), cvScalar(255,128,128));
			cvLine(outImage, cvPoint(tX, tY), cvPoint(vX, vY), cvScalar(128,255,128));
			cvShowImage("Training", outImage);
			cvReleaseImage(&outImage);
		} else {
			cvShowImage("Training", trainedDepthImage);
		}

		//Classify Stored
		{
			std::vector<ImagePoint> imagePoints;
			getAllImagePoints(storedDepthImage, imagePoints);

			predict(root, storedDepthImage, imagePoints, scale);

			IplImage *classImage = cvCreateImage(cvGetSize(storedDepthImage), IPL_DEPTH_8U, 3); cvSetZero(classImage);
			for (int i=0; i<imagePoints.size(); i++) {
				CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3) = ColorMap[imagePoints.at(i)._class].b; 
				CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3+1) = ColorMap[imagePoints.at(i)._class].g;
				CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3+2) = ColorMap[imagePoints.at(i)._class].r;
			}

			cvShowImage("Classified Stored Image", classImage);
			cvReleaseImage(&classImage);
		}

		//Classify Trained
		{
			std::vector<ImagePoint> imagePoints;
			getAllImagePoints(trainedDepthImage, imagePoints);

			predict(root, trainedDepthImage, imagePoints, 0.005);

			IplImage *classImage = cvCreateImage(cvGetSize(trainedDepthImage), IPL_DEPTH_8U, 3); cvSetZero(classImage);
			for (int i=0; i<imagePoints.size(); i++) {
				CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3) = ColorMap[imagePoints.at(i)._class].b; 
				CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3+1) = ColorMap[imagePoints.at(i)._class].g;
				CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3+2) = ColorMap[imagePoints.at(i)._class].r;
			}

			cvShowImage("Classified Trained Image", classImage);
			cvReleaseImage(&classImage);
		}
		
		switch (cvWaitKey(1)) {
			case '+':
			case '=':
				scale+=0.01;
				break;
			case '-':
				scale-=0.01;
				break;
			case 27:
				running = false;
		}
	}
}


IplImage* getDepthMask(IplImage *depth) {
	IplImage *depthMask = cvCreateImage(cvGetSize(depth), IPL_DEPTH_8U, 1); cvZero(depthMask);
	for (int y=0; y<depth->height; y++) for (int x=0; x<depth->width; x++) 
		if (CV_IMAGE_ELEM(depth, float, y, x)!=0) CV_IMAGE_ELEM(depthMask, unsigned char, y, x) = 255;
	return depthMask;
}

void showScaledDepth(const char* windowName, IplImage *depthImage) {
	//The Maximum and Minimum Kinect Depth values (used for visualisation)
	//float kinectDepthMax = 8000, kinectDepthMin = 400;
	double kinectDepthMax, kinectDepthMin; cvMinMaxLoc(depthImage, &kinectDepthMin, &kinectDepthMax);

	//Create a mask of the valid values for the depth image
	IplImage *depthImageMask = getDepthMask(depthImage);

	// Convert Depth Image into visible spectrum
	float scale = 255.0/(kinectDepthMax-kinectDepthMin), shift = -kinectDepthMin*scale;
	IplImage *depthImageVisible = cvCreateImage(cvGetSize(depthImage), IPL_DEPTH_8U, 1); cvSetZero(depthImageVisible);
	cvConvertScale(depthImage, depthImageVisible, scale, shift); cvSubRS(depthImageVisible, cvScalarAll(255), depthImageVisible, depthImageMask);
	cvShowImage(windowName, depthImageVisible);
	cvReleaseImage(&depthImageVisible);
	cvReleaseImage(&depthImageMask);
}

void getAllImagePoints(IplImage *depthIm, std::vector<ImagePoint> &ImagePoints) {
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

void predict(RandomTree *root, IplImage *depthIm, std::vector<ImagePoint> &imagePoints, float scale) {

	for (unsigned int i=0; i<imagePoints.size(); i++) {

		RandomTree *curNode = root;

		while (curNode->left!=0 || curNode->right!=0) {
			double value = GetFeatureValue(depthIm, imagePoints.at(i).x, imagePoints.at(i).y, curNode->splitFeature, scale);
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

double GetFeatureValue(IplImage* depthIm, int x, int y, Feature feature, float scale) {
	//Calculate the vector position
	float dix = 1.0/(CV_IMAGE_ELEM(depthIm, float, y, x)*scale);
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

#endif
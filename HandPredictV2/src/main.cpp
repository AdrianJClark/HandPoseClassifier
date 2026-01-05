#if 0
#include <XnOS.h>
#include <XnCppWrapper.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <windows.h>

#include "Features.h"
#include "RandomTree.h"
#include "ColorConvert.h"

struct ImagePoint {
	int frame, x, y, _class;
};

using namespace std;

int frameBegin=0; int frameEnd=60;

void calculateImagePoints(IplImage *depthIm, vector<ImagePoint> &ImagePoints);
void getAllImagePoints(IplImage *depthIm, vector<ImagePoint> &ImagePoints);
void predict(RandomTree *root, IplImage *depthIm, vector<ImagePoint> &ImagePoints);
double GetFeatureValue(IplImage* depthIm, int x, int y, Feature feature);
IplImage* getDepthMask(IplImage *depth);
IplImage *createKinectDepthMask(IplImage *depthImage);
void showScaledDepth(const char* windowName, IplImage *depthImage);
void colorMouseCallBack(int, int, int, int, void*);

using namespace xn;

std::vector<CvPoint> regionToExamine;

void main() {

	srand ( (unsigned int) time(NULL) );

	loadColorTable("Data/ColorTable.txt");
	RandomTree *root = loadTree("Data/TranslationTree.txt");

	//Kinect Objects
	Context niContext;
	DepthGenerator niDepth;
	ImageGenerator niImage;

	//Initialize Kinect
	EnumerationErrors errors;
	switch (XnStatus rc = niContext.InitFromXmlFile("SamplesConfig.xml", &errors)) {
		case XN_STATUS_OK:
			break;
		case XN_STATUS_NO_NODE_PRESENT:
			XnChar strError[1024];	errors.ToString(strError, 1024);
			printf("%s\n", strError);
			return; break;
		default:
			printf("Open failed: %s\n", xnGetStatusString(rc));
			return;
	}

	//Extract the Image and Depth nodes from the Kinect Context
	niContext.FindExistingNode(XN_NODE_TYPE_DEPTH, niDepth);
	niContext.FindExistingNode(XN_NODE_TYPE_IMAGE, niImage);

	//Align the depth image and colourImage
	niDepth.GetAlternativeViewPointCap().SetViewPoint(niImage);
	niDepth.GetMirrorCap().SetMirror(false);
	//niImage.GetMirrorCap().SetMirror(false);

	IplImage *averageDepth = 0; 

	bool running = true;
	while (running) {

		if (XnStatus rc = niContext.WaitAnyUpdateAll() != XN_STATUS_OK) {
			printf("Read failed: %s\n", xnGetStatusString(rc));
			return;
		}

		// Update MetaData containers
		DepthMetaData niDepthMD; ImageMetaData niImageMD;
		niDepth.GetMetaData(niDepthMD); niImage.GetMetaData(niImageMD);

		// Extract Colour Image
		IplImage *colourImage = cvCreateImage(cvSize(niImageMD.XRes(), niImageMD.YRes()), IPL_DEPTH_8U, 3);
		memcpy(colourImage->imageData, niImageMD.Data(), colourImage->imageSize); cvConvertImage(colourImage, colourImage, CV_CVTIMG_SWAP_RB);
		cvFlip(colourImage, colourImage, 1);

		IplImage *colourClone = cvCloneImage(colourImage);
		for (int i=0; i< regionToExamine.size(); i++)
			cvCircle(colourClone, regionToExamine.at(i), 3, cvScalar(255,0,255));
		cvSetMouseCallback("Colour Image", colorMouseCallBack);
		cvShowImage("Colour Image", colourClone);
		cvReleaseImage(&colourClone);

		int x1=colourImage->width, y1=colourImage->height, x2=0, y2=0;
		if (regionToExamine.size()>3) {
			for (int i=0; i< regionToExamine.size(); i++) {
				if (regionToExamine.at(i).x < x1) x1 = regionToExamine.at(i).x;
				if (regionToExamine.at(i).x > x2) x2 = regionToExamine.at(i).x;
				if (regionToExamine.at(i).x < y1) y1 = regionToExamine.at(i).y;
				if (regionToExamine.at(i).x > y2) y2 = regionToExamine.at(i).y;
			}

			IplImage *croppedImage = cvCreateImage(cvSize(x2-x1, y2-y1), IPL_DEPTH_8U, 3);
			cvSetImageROI(colourImage, cvRect(x1, y1, x2-x1, y2-y1)); cvCopy(colourImage, croppedImage);
			cvResetImageROI(colourImage);
			cvShowImage("CroppedImage", croppedImage);
			cvReleaseImage(&croppedImage);
		}


		// Extract Depth Image
		IplImage *depthImage = cvCreateImage(cvSize(niImageMD.XRes(), niImageMD.YRes()), IPL_DEPTH_16U, 1);
		memcpy(depthImage->imageData, niDepthMD.Data(), depthImage->imageSize);
		IplImage *depthFloatImage = cvCreateImage(cvGetSize(depthImage), IPL_DEPTH_32F, 1);
		cvConvertScale(depthImage, depthFloatImage);
		showScaledDepth("Depth Visible", depthFloatImage);

		IplImage *diCroppedImage=0;
		if (averageDepth!=0) {
			showScaledDepth("Average Depth", averageDepth);

			IplImage *aboveDepth = cvCreateImage(cvGetSize(depthFloatImage), IPL_DEPTH_32F, 1);
			IplImage *aboveDepthMask = cvCreateImage(cvGetSize(depthFloatImage), IPL_DEPTH_8U, 1); cvSetZero(aboveDepthMask);

			cvSub(averageDepth, depthFloatImage, aboveDepth);
			for (int y=0; y<aboveDepth->height; y++) for (int x=0; x<aboveDepth->width; x++) 
				if (CV_IMAGE_ELEM(aboveDepth, float, y, x)>20) CV_IMAGE_ELEM(aboveDepthMask, unsigned char, y, x)=255;

			showScaledDepth("Above Depth", aboveDepth);

			cvShowImage("aboveDepthMask", aboveDepthMask);

			if (regionToExamine.size()>3) {
				//Depth Mask
				IplImage *dmCroppedImage = cvCreateImage(cvSize(x2-x1, y2-y1), IPL_DEPTH_8U, 1);
				cvSetImageROI(aboveDepthMask, cvRect(x1, y1, x2-x1, y2-y1)); cvCopy(aboveDepthMask, dmCroppedImage);
				cvResetImageROI(aboveDepthMask);
				cvShowImage("Cropped Depth Mask", dmCroppedImage);
				cvNot(dmCroppedImage, dmCroppedImage);
			
				//Depth Image
				diCroppedImage = cvCreateImage(cvSize(x2-x1, y2-y1), IPL_DEPTH_32F, 1);
				cvSetImageROI(aboveDepth, cvRect(x1, y1, x2-x1, y2-y1)); cvCopy(aboveDepth, diCroppedImage);
				cvResetImageROI(aboveDepth);
				cvSet(diCroppedImage, cvScalarAll(0), dmCroppedImage);
				cvMirror(diCroppedImage, diCroppedImage, 1);
				showScaledDepth("Cropped Depth", diCroppedImage);

				vector<ImagePoint> imagePoints;
				getAllImagePoints(diCroppedImage, imagePoints);
	
				predict(root, diCroppedImage, imagePoints);

				IplImage *classImage = cvCreateImage(cvGetSize(diCroppedImage), IPL_DEPTH_8U, 3); cvSetZero(classImage);
				for (int i=0; i<imagePoints.size(); i++) {
					CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3) = ColorMap[imagePoints.at(i)._class].b; 
					CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3+1) = ColorMap[imagePoints.at(i)._class].g;
					CV_IMAGE_ELEM(classImage, unsigned char, imagePoints.at(i).y, imagePoints.at(i).x*3+2) = ColorMap[imagePoints.at(i)._class].r;
				}

				cvShowImage("Classified Image", classImage);

				cvReleaseImage(&classImage);

				cvReleaseImage(&dmCroppedImage);

			}

			cvReleaseImage(&aboveDepth);
			cvReleaseImage(&aboveDepthMask);
		}

		

		switch (cvWaitKey(1)) {
			case 27: running = false; break;
			case ' ':
				if (averageDepth!=0) cvReleaseImage(&averageDepth);
				averageDepth = cvCloneImage(depthFloatImage);
				break;
			case 13:
				if (diCroppedImage!=0) {
					CvFileStorage* fstorage = cvOpenFileStorage("croppedDepth.yml", NULL, CV_STORAGE_WRITE);
					cvWrite( fstorage, "depth", diCroppedImage );
					cvReleaseFileStorage(&fstorage);
				}
				break;
		}

		if (diCroppedImage!=0) cvReleaseImage(&diCroppedImage);
		cvReleaseImage(&depthFloatImage);	
		cvReleaseImage(&depthImage);
		cvReleaseImage(&colourImage);


	}
	cvReleaseImage(&averageDepth);

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
	float dix = 1.0/(CV_IMAGE_ELEM(depthIm, float, y, x)*0.04);
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


//Create a mask of valid depth values
IplImage *createKinectDepthMask(IplImage *depthImage) {
	IplImage *depthImageMask = cvCreateImage(cvGetSize(depthImage), IPL_DEPTH_8U, 1); cvSetZero(depthImageMask);
	//Loop through each pixel in the depth image, if the depth value isn't 0, set the corresponding pixel in the mask to 255
	for (int y=0; y<depthImage->height;y++) for (int x=0; x<depthImage->width; x++)
		if (CV_IMAGE_ELEM(depthImage, unsigned short, y, x)!=0) CV_IMAGE_ELEM(depthImageMask, unsigned char, y, x)=255;

	return depthImageMask;
}

void showScaledDepth(const char* windowName, IplImage *depthImage) {
	//The Maximum and Minimum Kinect Depth values (used for visualisation)
	float kinectDepthMax = 8000, kinectDepthMin = 400;

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

void colorMouseCallBack(int _event, int x, int y, int flags, void *p) {
	if (_event == CV_EVENT_LBUTTONDOWN) {
		regionToExamine.push_back(cvPoint(x, y));	
	} else if (_event == CV_EVENT_RBUTTONDOWN) {
		regionToExamine.clear();
	}
}
#endif
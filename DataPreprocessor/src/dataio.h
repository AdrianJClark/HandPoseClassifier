#ifndef DATAIO_H
#define DATAIO_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "Settings.h"

using namespace std;

bool GrabFrame(IplImage **rgb, IplImage **depth, int index) {

	//Check if the input is a video or collection of videos, and load appropriately
	if (colorInPath.substr(colorInPath.find_last_of("."))==".avi") {
		CvCapture *cap = cvCreateFileCapture(colorInPath.c_str());
		cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, index);
		(*rgb) = cvCloneImage(cvQueryFrame(cap));
		cvReleaseCapture(&cap);
	} else {
		char filename[100]; sprintf(filename, colorInPath.c_str(), index);
		(*rgb) = cvLoadImage(filename);
	}

	char filename[100]; sprintf(filename, depthInPath.c_str(), index);
	FILE *f = fopen(filename, "rb");
	if (f==0) return false;

	//Inches to centimeters
	float scale = 25.4;
	(*depth) = cvCreateImage(cvSize(320,240), IPL_DEPTH_32F, 1); cvZero(*depth);
	for (int y = 0; y<240; y++) {
		for (int x=0; x<320; x++) {
			float d; fscanf(f, "%f\n", &d);
			if (d!=float(-1e+030)) {
				CV_IMAGE_ELEM(*depth, float, y, x) = -d * scale;
			}
		}
	}

	fclose(f);
	return true;

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

IplImage* getColorMask(IplImage *color) {
	IplImage *colorMask = cvCreateImage(cvGetSize(color), IPL_DEPTH_8U, 1); cvZero(colorMask);

	for (int y=0; y<color->height; y++) {
		for (int x=0; x<color->width; x++) {
			if (!(CV_IMAGE_ELEM(color, unsigned char, y, x*3)==0 && CV_IMAGE_ELEM(color, unsigned char, y, x*3+1)==0 && CV_IMAGE_ELEM(color, unsigned char, y, x*3+2)==0)) 
				CV_IMAGE_ELEM(colorMask, unsigned char, y, x) = 255;
		}
	}

	return colorMask;
}

#endif

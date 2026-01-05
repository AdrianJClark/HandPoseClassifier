#ifndef DATAIO_H
#define DATAIO_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
using namespace std;

CvCapture *cRGB;
char *fileDepth;
int frameNumber;


void loadData(char *_fileColor, char *_fileDepth) {

	cRGB = cvCreateFileCapture(_fileColor);
	fileDepth = (char*)calloc(strlen(_fileDepth)+1, sizeof(char));
	strncpy(fileDepth, _fileDepth, strlen(_fileDepth));
	frameNumber = 0;
	
}

bool GrabFrame(IplImage **rgb, IplImage **depth) {

	IplImage *fRGB = cvQueryFrame(cRGB);
	if (fRGB==0) return false;

	(*rgb) = cvCloneImage(fRGB);


	float scale = 25.4;
	float _void; sscanf("-1e+030", "%f", &_void);
	char filename[50]; sprintf(filename, fileDepth, frameNumber);
	FILE *f = fopen(filename, "rb");
	if (f==0) return false;

	(*depth) = cvCreateImage(cvSize(320,240), IPL_DEPTH_32F, 1); cvZero(*depth);
	for (int y = 0; y<240; y++) {
		for (int x=0; x<320; x++) {
			float d; fscanf(f, "%f\n", &d);
			if (d!=_void) {
				CV_IMAGE_ELEM(*depth, float, y, x) = -d * scale;
			}
		}
	}

	frameNumber++;

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

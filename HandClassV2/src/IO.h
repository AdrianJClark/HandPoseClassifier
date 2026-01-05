#ifndef IO_H
#define IO_H

#include <string>

IplImage *getClassImage(int frame, const char* classPath) {
	char filename[50]; sprintf(filename, classPath, frame);
	return cvLoadImage(filename, 0);
}

IplImage *getDepthImage(int frame, const char* depthPath) {
	char filename[50]; sprintf(filename, depthPath, frame);
	CvFileStorage* fs = cvOpenFileStorage( filename, 0, CV_STORAGE_READ );
	IplImage *depth = (IplImage*)cvRead(fs, cvGetFileNodeByName( fs, NULL, "depth" ));
	cvReleaseFileStorage(&fs);
	return depth;
}

IplImage* getMask(IplImage *image) {
	IplImage *imageMask = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1); cvZero(imageMask);

	for (int y=0; y<image->height; y++) {
		for (int x=0; x<image->width; x++) {
			if (!(CV_IMAGE_ELEM(image, unsigned char, y, x*3)==0 && CV_IMAGE_ELEM(image, unsigned char, y, x*3+1)==0 && CV_IMAGE_ELEM(image, unsigned char, y, x*3+2)==0)) 
				CV_IMAGE_ELEM(imageMask, unsigned char, y, x) = 255;
		}
	}

	return imageMask;
}

#endif
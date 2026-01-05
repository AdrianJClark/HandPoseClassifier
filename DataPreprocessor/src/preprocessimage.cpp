#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/features2d/features2d.hpp>

#include <time.h>
#include <windows.h>

#include "Settings.h"
#include "dataio.h"
#include "colormatch.h"

using namespace std;

void main() {

	loadSettings("settings.xml");
	loadColorTable(colorTable.c_str());
	
	//Make sure output directories exist
	CreateDirectory(colorOutPath.substr(0, colorOutPath.find_last_of("/")).c_str(), NULL);
	CreateDirectory(classOutPath.substr(0, classOutPath.find_last_of("/")).c_str(), NULL);
	CreateDirectory(depthOutPath.substr(0, depthOutPath.find_last_of("/")).c_str(), NULL);

	for (int i=frameBegin; i<=frameEnd; i++) {
	
		IplImage *rgb, *depth;
		if (!GrabFrame(&rgb, &depth, i)) break;

		IplImage *rgbS = cvCreateImage(cvSize(160, 120), rgb->depth, rgb->nChannels);
		IplImage *depthS = cvCreateImage(cvSize(160, 120), depth->depth, depth->nChannels);

		cv::resize(cv::Mat(rgb), cv::Mat(rgbS), cv::Size(160,120));
		cv::resize(cv::Mat(depth), cv::Mat(depthS), cv::Size(160,120));

		
		IplImage *visible = cvCreateImage(cvGetSize(depth), IPL_DEPTH_8U, 1);
		double min, max; cvMinMaxLoc(depth, &min, &max);
		cvConvertScale(depth, visible, 255.0/(max-min));
		cvShowImage("visible", visible);
		cvReleaseImage(&visible);
		

		IplImage *colMatched = colorMatchImage(rgbS);
		cvShowImage("colMatched", colMatched); cvWaitKey(1);
		IplImage *colClass = colorClassifyImage(rgbS);

		printf("Processing: %d%%\r", int(double(i)/double(frameEnd)*100.0));

		char filename[50]; 
		sprintf(filename,colorOutPath.c_str(), i);
		cvSaveImage(filename, colMatched);

		sprintf(filename, classOutPath.c_str(), i);
		cvSaveImage(filename, colClass);

		sprintf(filename, depthOutPath.c_str(), i);
		CvFileStorage* fstorage = cvOpenFileStorage(filename, NULL, CV_STORAGE_WRITE);
	    cvWrite( fstorage, "depth", depthS );
		cvReleaseFileStorage(&fstorage);

		cvReleaseImage(&colMatched);

		switch (cvWaitKey(1)) {
			case 27:
				exit(0);
				break;
		}

		cvReleaseImage(&rgbS);
		cvReleaseImage(&depthS);

		cvReleaseImage(&rgb);
		cvReleaseImage(&depth);
	};

}
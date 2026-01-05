#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/features2d/features2d.hpp>

#include <time.h>

#include "dataio.h"
#include "features.h"
#include "colormatch.h"

using namespace std;
void ProcessFrame(IplImage* rgbS, IplImage* depthS, vector<Feature> featureList, cv::Mat &classes, cv::Mat &descriptors);

void main() {

	vector<Feature> featureList = loadUV("features.txt");
	loadData("data/color_translation.avi", "data/Translation%d.txt");
	initColorMatcher();

	cv::Mat classes = cv::Mat(0, 1, CV_8UC1);
	cv::Mat descriptors = cv::Mat(0, featureList.size(), CV_32FC1);
	
	bool running = true;
	while (running) {
		IplImage *rgb, *depth;
		if (!GrabFrame(&rgb, &depth)) break;

		IplImage *depthMask = getDepthMask(depth);
		cvShowImage("Depth Mask", depthMask);

		IplImage *depthVisible = cvCreateImage(cvGetSize(depth), IPL_DEPTH_8U, 1);
		double scale = 255.0/(500.0-180.0); double shift = -180*scale;
		cvConvertScale(depth, depthVisible, scale, shift);
		cvSubRS(depthVisible, cvScalar(255), depthVisible, depthMask);
		cvShowImage("Depth Visible", depthVisible);
		cvShowImage("rgb", rgb);

		IplImage *rgbS = cvCreateImage(cvSize(160, 120), rgb->depth, rgb->nChannels);
		IplImage *depthS = cvCreateImage(cvSize(160, 120), depth->depth, depth->nChannels);

		cv::resize(cv::Mat(rgb), cv::Mat(rgbS), cv::Size(160,120));
		cv::resize(cv::Mat(depth), cv::Mat(depthS), cv::Size(160,120));
		
		ProcessFrame(rgbS, depthS, featureList, classes, descriptors);

		printf("%d, %d\n", classes.rows, descriptors.rows);
		cvShowImage("rgbs", rgbS);
		cvShowImage("depths", depthS);

		switch (cvWaitKey(1)) {
			case 27:
				running = false;
				break;
		}

		cvReleaseImage(&rgbS);
		cvReleaseImage(&depthS);

		cvReleaseImage(&rgb);
		cvReleaseImage(&depth);
	};

}

void ProcessFrame(IplImage* rgbS, IplImage* depthS, vector<Feature> featureList, cv::Mat &classes, cv::Mat &descriptors) {

	IplImage *colMatched = colorMatchImage(rgbS);
	IplImage *colMask = getColorMask(colMatched);
	int nonZero = cvCountNonZero(colMask);
	cvShowImage("colMatched", colMatched);
	cvShowImage("colMask", colMask);

	int pMax = (rgbS->width*rgbS->height)/10;
	if (pMax>float(nonZero)*.9) pMax = float(nonZero)*.9;
	srand ( time(NULL) );

	cv::Mat trainDescriptors = cv::Mat(pMax, featureList.size(), CV_32FC1);
	cv::Mat query = cv::Mat(pMax, 3, CV_32FC1);

	int pCount=0;
	while (pCount<pMax) {
		
		int x = (double(rand())/double(RAND_MAX))*double(colMask->width-1);
		int y = (double(rand())/double(RAND_MAX))*double(colMask->height-1);

		if (CV_IMAGE_ELEM(colMask, unsigned char, y, x)!=0) {
			CV_IMAGE_ELEM(colMask, unsigned char, y, x)=0;

			float dix = CV_IMAGE_ELEM(depthS, float, y, x)/200.0;

			for (int i=0; i<featureList.size(); i++) {
				float uxd = featureList.at(i).uX/dix;
				float uyd = featureList.at(i).uY/dix;
				float vxd = featureList.at(i).vX/dix;
				float vyd = featureList.at(i).vY/dix;

				float uX = x + uxd, uY = y + uyd;
				float vX = x + vxd, vY = y + vyd;

				float diU, diV;
				if (uX<0 || uY<0 || uX>=depthS->width || uY>=depthS->height) {
					diU = 10000;
				} else {
					diU = CV_IMAGE_ELEM(depthS, float, (int)uY, (int)uX);
				}

				if (vX<0 || vY<0 || vX>=depthS->width || vY>=depthS->height) {
					diV = 10000;
				} else {
					diV = CV_IMAGE_ELEM(depthS, float, (int)vY, (int)vX);
				}

				if (diU==0.0) diU = 10000;
				if (diV==0.0) diV = 10000;
				CV_MAT_ELEM((CvMat)trainDescriptors, float, pCount, i) = diU-diV;


			}

			CV_MAT_ELEM((CvMat)query, float, pCount, 0) = CV_IMAGE_ELEM(colMatched, unsigned char, y, x*3);
			CV_MAT_ELEM((CvMat)query, float, pCount, 1) = CV_IMAGE_ELEM(colMatched, unsigned char, y, x*3+1);
			CV_MAT_ELEM((CvMat)query, float, pCount, 2) = CV_IMAGE_ELEM(colMatched, unsigned char, y, x*3+2);

			pCount++;
		}
		
	}

	cv::Mat trainClasses = colorMatchMat(query);
	cvReleaseImage(&colMatched);
	cvShowImage("pMask", colMask);

	classes.push_back(trainClasses);
	descriptors.push_back(trainDescriptors);

	cvReleaseImage(&colMask);
}
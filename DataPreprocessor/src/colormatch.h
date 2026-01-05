#ifndef COLORMATCH_H
#define COLORMATCH_H

#include <map>

cv::FlannBasedMatcher ColorMatcher;

struct ClassColor {
	int _class;
	int r, g, b;
};

std::map<int, ClassColor> ColorMap;

void initColorMatcher() {
	float *colourData = (float*)malloc(sizeof(float)*ColorMap.size()*3);
	int i=0;
	for (std::map<int,ClassColor>::iterator it=ColorMap.begin(); it!=ColorMap.end(); it++) {
		colourData[i*3] = it->second.b; colourData[i*3+1] = it->second.g; colourData[i*3+2] = it->second.r;
		i++;
	}
	cv::Mat matColor = cv::Mat(22, 3, CV_32FC1, colourData);
	vector<cv::Mat> colourDescriptors; colourDescriptors.push_back(matColor);
	ColorMatcher.add(colourDescriptors);
	ColorMatcher.train();
}

void loadColorTable(const char* filename) {
	FILE *f = fopen(filename, "rb");
	ClassColor c;
	while (fscanf(f, "%d=%d,%d,%d\r\n", &c._class, &c.r, &c.g, &c.b)==4) {
		ColorMap[c._class] = c;
	}
	initColorMatcher();
}

IplImage *colorMatchImage(IplImage *original, float thresh=3.0) {
	cv::Mat query = cv::Mat(original->width*original->height, 3, CV_32FC1);
	for (int y=0; y<original->height; y++) {
		for (int x=0; x<original->width; x++) {
			CV_MAT_ELEM((CvMat)query, float, x+(y*original->width), 0) = float(CV_IMAGE_ELEM(original, unsigned char, y, x*3));
			CV_MAT_ELEM((CvMat)query, float, x+(y*original->width), 1) = float(CV_IMAGE_ELEM(original, unsigned char, y, x*3+1));
			CV_MAT_ELEM((CvMat)query, float, x+(y*original->width), 2) = float(CV_IMAGE_ELEM(original, unsigned char, y, x*3+2));
		}
	}
	vector<cv::DMatch> matches;
	ColorMatcher.match(query, matches);

	IplImage *out = cvCreateImage(cvGetSize(original), IPL_DEPTH_8U, 3); cvZero(out);
	for (int y=0; y<out->height; y++) {
		for (int x=0; x<out->width; x++) {
			if (matches.at(x+(y*original->width)).distance < thresh) {
				int index = matches.at(x+(y*original->width)).trainIdx;
				CV_IMAGE_ELEM(out, unsigned char, y, x*3) = ColorMap[index].b;
				CV_IMAGE_ELEM(out, unsigned char, y, x*3+1) = ColorMap[index].g;
				CV_IMAGE_ELEM(out, unsigned char, y, x*3+2) = ColorMap[index].r;
			}
		}
	}

	return out;

}

IplImage *colorClassifyImage(IplImage *original, float thresh=3.0) {
	cv::Mat query = cv::Mat(original->width*original->height, 3, CV_32FC1);
	for (int y=0; y<original->height; y++) {
		for (int x=0; x<original->width; x++) {
			CV_MAT_ELEM((CvMat)query, float, x+(y*original->width), 0) = float(CV_IMAGE_ELEM(original, unsigned char, y, x*3));
			CV_MAT_ELEM((CvMat)query, float, x+(y*original->width), 1) = float(CV_IMAGE_ELEM(original, unsigned char, y, x*3+1));
			CV_MAT_ELEM((CvMat)query, float, x+(y*original->width), 2) = float(CV_IMAGE_ELEM(original, unsigned char, y, x*3+2));
		}
	}
	vector<cv::DMatch> matches;
	ColorMatcher.match(query, matches);

	IplImage *out = cvCreateImage(cvGetSize(original), IPL_DEPTH_8U, 1); cvZero(out);
	for (int y=0; y<out->height; y++) {
		for (int x=0; x<out->width; x++) {
			if (matches.at(x+(y*original->width)).distance < thresh) {
				CV_IMAGE_ELEM(out, unsigned char, y, x) = matches.at(x+(y*original->width)).trainIdx;
			}
		}
	}
	return out;
}

cv::Mat colorMatchMat(cv::Mat query) {

	vector<cv::DMatch> matches;
	ColorMatcher.match(query, matches);

	cv::Mat out = cv::Mat(matches.size(), 1, CV_8UC1);
	for (int i=0; i<matches.size(); i++)
		CV_MAT_ELEM((CvMat)out, unsigned char, i, 0) = matches.at(i).trainIdx;

	return out;
}


#endif
#ifndef COLORMATCH_H
#define COLORMATCH_H

cv::FlannBasedMatcher ColorMatcher;

float colourData[] = {
	0,0,0,
	105, 105, 105,
	254, 254, 254,
	106, 69, 12,
	12,105,35,
	12,12,105,
	0,80,254,
	106,105,254,
	35,35,254,
	0,0,254,
	106,254,0,
	105,254,163,
	0,254,55,
	0,230,254,
	254,0,18,
	254,230,106,
	254,163,0,
	125,196,66,
	104,0,107,
	254,107,254,
	254,0,254,
	254,105,133
};

void initColorMatcher() {
	cv::Mat matColor = cv::Mat(22, 3, CV_32FC1, colourData);
	vector<cv::Mat> colourDescriptors; colourDescriptors.push_back(matColor);
	ColorMatcher.add(colourDescriptors);
	ColorMatcher.train();
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
				CV_IMAGE_ELEM(out, unsigned char, y, x*3) = colourData[index*3];
				CV_IMAGE_ELEM(out, unsigned char, y, x*3+1) = colourData[index*3+1];
				CV_IMAGE_ELEM(out, unsigned char, y, x*3+2) = colourData[index*3+2];
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
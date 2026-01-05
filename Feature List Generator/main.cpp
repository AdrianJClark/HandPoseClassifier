#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>

using namespace std;

void generateUV();
void loadUV(char *filename);

struct Feature {
	int uX, uY;
	int vX, vY;
};

void main() {

	generateUV();
	//loadUV("features.txt");

	return;

	CvCapture *cRGB = cvCreateFileCapture("hand_default_rgb.avi");
	CvCapture *cDepth = cvCreateFileCapture("hand_default_d.avi");


	bool running = true;
	while (running) {
		cvGrabFrame(cRGB); cvGrabFrame(cDepth); 
		IplImage *rgb = cvCloneImage(cvRetrieveFrame(cRGB));
		IplImage *depth = cvCloneImage(cvRetrieveFrame(cDepth));

		cvShowImage("rgb", rgb);
		cvShowImage("depth", depth);

		switch (cvWaitKey(1)) {
			case 27:
				running = false;
				break;
		}

		cvReleaseImage(&rgb);
		cvReleaseImage(&depth);
	};

}


void generateUV() {

	vector<Feature> featureList;

	srand ( time(NULL) );

	int misses=0;
	while (featureList.size()<4000) {
		Feature f;
		{
			float ang = float(rand())/float(RAND_MAX)*360.0;
			float dist = float(rand())/float(RAND_MAX)*60.0;
			float angRad = (ang-90) * 0.01745329251994329576923690768489;
			f.uX = cos(angRad)*dist;
			f.uY = sin(angRad)*dist;
		}

		{
			float ang = float(rand())/float(RAND_MAX)*360.0;
			float dist = float(rand())/float(RAND_MAX)*60.0;
			float angRad = (ang-90) * 0.01745329251994329576923690768489;
			f.vX = cos(angRad)*dist;
			f.vY = sin(angRad)*dist;
		}

		bool tooClose = false; float thresh = 5;
		for (int j=0; j<featureList.size(); j++) {
			float distV = sqrt(float((featureList.at(j).vX - f.vX) * (featureList.at(j).vX - f.vX) + 
									 (featureList.at(j).vY - f.vY) * (featureList.at(j).vY - f.vY)));
			float distU = sqrt(float((featureList.at(j).uX - f.uX) * (featureList.at(j).uX - f.uX) + 
									 (featureList.at(j).uY - f.uY) * (featureList.at(j).uY - f.uY)));

			if (distV<thresh && distU<thresh) {
				tooClose = true; misses++; break;
			}
		}

		if (!tooClose) featureList.push_back(f);
		printf("Generating Feature List: %d%%\r", int((float)featureList.size()/4000.0*100.0));
	}
	
	printf("Number of Calculated Features: %d. Misses: %d\n", featureList.size(), misses);

	IplImage *testImage = cvCreateImage(cvSize(120,120), IPL_DEPTH_8U, 3); cvSet(testImage, cvScalarAll(255));

	for (int i=0; i<featureList.size(); i++) {
		cvLine(testImage, cvPoint(60,60), cvPoint(60+featureList.at(i).uX, 60+featureList.at(i).uY), cvScalar(255), 1);
		cvLine(testImage, cvPoint(60,60), cvPoint(60+featureList.at(i).vX, 60+featureList.at(i).vY), cvScalar(0,0,255), 1);
	}
	cvShowImage("angle", testImage);
	cvWaitKey();
	cvReleaseImage(&testImage);

	FILE *f = fopen("features.txt", "wb");
	fprintf(f, "%d\r\n", featureList.size());
	for (int i=0; i<featureList.size(); i++) {
		fprintf(f, "%d\t%d\t%d\t%d\r\n", 
			featureList.at(i).uX, featureList.at(i).uY,
			featureList.at(i).vX, featureList.at(i).vY);
	}
	fclose(f);
}

void loadUV(char *filename) {

	vector<Feature> featureList;


	FILE *f = fopen(filename, "rb");
	int count;
	fscanf(f, "%d\r\n", &count);
	
	for (int i=0; i<count; i++) {
		Feature fe;
		fscanf(f, "%d\t%d\t%d\t%d\r\n", &fe.uX, &fe.uY, &fe.vX, &fe.vY);
		featureList.push_back(fe);
	}
	fclose(f);


	IplImage *testImage = cvCreateImage(cvSize(120,120), IPL_DEPTH_8U, 3); cvSet(testImage, cvScalarAll(255));
	for (int i=0; i<featureList.size(); i++) {
		cvLine(testImage, cvPoint(60,60), cvPoint(60+featureList.at(i).uX, 60+featureList.at(i).uY), cvScalar(255), 1);
		cvLine(testImage, cvPoint(60,60), cvPoint(60+featureList.at(i).vX, 60+featureList.at(i).vY), cvScalar(0,0,255), 1);
	}
	cvShowImage("angle", testImage);
	cvWaitKey();
	cvReleaseImage(&testImage);

}

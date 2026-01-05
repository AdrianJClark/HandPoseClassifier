#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <math.h>
#include <time.h>

struct Feature {
	int uX, uY;
	int vX, vY;
	double threshold;
};

struct CalculatedFeature {
	int _class;
	double value;
};

//void renderFeatures(std::vector<Feature> featureList);

void generateUV(char *filename) {

	std::vector<Feature> featureList;

	srand ( (unsigned int)time(NULL) );

	int misses=0;
	while (featureList.size()<4000) {
		Feature f;
		{
			double ang = double(rand())/double(RAND_MAX)*360.0;
			double dist = double(rand())/double(RAND_MAX)*60.0;
			double angRad = (ang-90) * 0.01745329251994329576923690768489;
			f.uX = int(cos(angRad)*dist);
			f.uY = int(sin(angRad)*dist);
		}

		{
			double ang = double(rand())/double(RAND_MAX)*360.0;
			double dist = double(rand())/double(RAND_MAX)*60.0;
			double angRad = (ang-90) * 0.01745329251994329576923690768489;
			f.vX = int(cos(angRad)*dist);
			f.vY = int(sin(angRad)*dist);
		}

		{
			double thresh = double(rand())/double(RAND_MAX)*200.0;
			f.threshold = thresh-100.0;
		}

		bool tooClose = false; double thresh = 5;
		for (unsigned int j=0; j<featureList.size(); j++) {
			double distV = sqrt(double((featureList.at(j).vX - f.vX) * (featureList.at(j).vX - f.vX) + 
									 (featureList.at(j).vY - f.vY) * (featureList.at(j).vY - f.vY)));
			double distU = sqrt(double((featureList.at(j).uX - f.uX) * (featureList.at(j).uX - f.uX) + 
									 (featureList.at(j).uY - f.uY) * (featureList.at(j).uY - f.uY)));

			if (distV<thresh && distU<thresh) {
				tooClose = true; misses++; break;
			}
		}

		if (!tooClose) featureList.push_back(f);
		printf("Generating Feature List: %d%%\r", int((double)featureList.size()/4000.0*100.0));
	}
	
	printf("Number of Calculated Features: %d. Misses: %d\n", featureList.size(), misses);

	//renderFeatures(featureList);

	FILE *f = fopen(filename, "wb");
	fprintf(f, "%d\r\n", featureList.size());
	for (unsigned int i=0; i<featureList.size(); i++) {
		fprintf(f, "%d\t%d\t%d\t%d\t%lf\r\n", 
			featureList.at(i).uX, featureList.at(i).uY,
			featureList.at(i).vX, featureList.at(i).vY,
			featureList.at(i).threshold);
	}
	fclose(f);
}

std::vector<Feature> loadUV(const char *filename) {

	std::vector<Feature> featureList;

	FILE *f = fopen(filename, "rb");
	int count;
	fscanf(f, "%d\r\n", &count);
	
	for (int i=0; i<count; i++) {
		Feature fe;
		fscanf(f, "%d\t%d\t%d\t%d\t%lf\r\n", &fe.uX, &fe.uY, &fe.vX, &fe.vY, &fe.threshold);
		featureList.push_back(fe);
	}
	fclose(f);

	return featureList;
}

/*void renderFeatures(std::vector<Feature> featureList) {
	IplImage *testImage = cvCreateImage(cvSize(120,120), IPL_DEPTH_8U, 3); cvSet(testImage, cvScalarAll(255));
	for (unsigned int i=0; i<featureList.size(); i++) {
		cvLine(testImage, cvPoint(60,60), cvPoint(60+featureList.at(i).uX, 60+featureList.at(i).uY), cvScalar(255), 1);
		cvLine(testImage, cvPoint(60,60), cvPoint(60+featureList.at(i).vX, 60+featureList.at(i).vY), cvScalar(0,0,255), 1);
	}
	cvShowImage("angle", testImage);
	cvWaitKey(1);
	cvReleaseImage(&testImage);

}*/

#endif
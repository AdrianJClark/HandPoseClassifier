#ifndef FEATURES_H
#define FEATURES_H

struct Feature {
	int uX, uY;
	int vX, vY;
};


void renderFeatures(vector<Feature> featureList);

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

	renderFeatures(featureList);

	FILE *f = fopen("features.txt", "wb");
	fprintf(f, "%d\r\n", featureList.size());
	for (int i=0; i<featureList.size(); i++) {
		fprintf(f, "%d\t%d\t%d\t%d\r\n", 
			featureList.at(i).uX, featureList.at(i).uY,
			featureList.at(i).vX, featureList.at(i).vY);
	}
	fclose(f);
}

vector<Feature> loadUV(char *filename) {

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

	renderFeatures(featureList);
	return featureList;
}

void renderFeatures(vector<Feature> featureList) {
	IplImage *testImage = cvCreateImage(cvSize(120,120), IPL_DEPTH_8U, 3); cvSet(testImage, cvScalarAll(255));
	for (int i=0; i<featureList.size(); i++) {
		cvLine(testImage, cvPoint(60,60), cvPoint(60+featureList.at(i).uX, 60+featureList.at(i).uY), cvScalar(255), 1);
		cvLine(testImage, cvPoint(60,60), cvPoint(60+featureList.at(i).vX, 60+featureList.at(i).vY), cvScalar(0,0,255), 1);
	}
	cvShowImage("angle", testImage);
	cvWaitKey(1);
	cvReleaseImage(&testImage);

}

#endif
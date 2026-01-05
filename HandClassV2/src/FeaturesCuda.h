#ifndef FEATURESCUDA_H
#define FEATURESCUDA_H

#include <cuda_runtime.h>
#include <vector_types.h>

#include <vector>
#include <math.h>
#include <time.h>

int* FeaturesUX, *FeaturesUY;
int* FeaturesVX, *FeaturesVY;
double* FeaturesThresh;
int FeaturesCount;

void loadUVCuda(const char *filename) {

	FILE *f = fopen(filename, "rb");
	fscanf(f, "%d\r\n", &FeaturesCount);
	
	FeaturesUX = (int*)malloc(FeaturesCount*sizeof(int));
	FeaturesUY = (int*)malloc(FeaturesCount*sizeof(int));
	FeaturesVX = (int*)malloc(FeaturesCount*sizeof(int));
	FeaturesVY = (int*)malloc(FeaturesCount*sizeof(int));
	FeaturesThresh = (double*)malloc(FeaturesCount*sizeof(double));

	for (int i=0; i<FeaturesCount; i++) {
		int* fUX = FeaturesUX+i, *fUY = FeaturesUY+i, *fVX = FeaturesVX+i, *fVY = FeaturesVY+i;
		double *fT = FeaturesThresh+i;

		fscanf(f, "%d\t%d\t%d\t%d\t%lf\r\n", fUX, fUY, fVX, fVY, fT);
	}
	fclose(f);

}


#endif
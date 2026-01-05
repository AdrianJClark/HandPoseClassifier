/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example of integrating CUDA functions into an existing 
 * application / framework.
 * Device code.
 */

#ifndef _CPP_INTEGRATION_KERNEL_H_
#define _CPP_INTEGRATION_KERNEL_H_

///////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_odata  memory to process (in and out)
///////////////////////////////////////////////////////////////////////////////
__global__ void
kernel_get_feature_values(int threadPerBlock, int2 *cudaImagePointsPos, int2 *cudaImagePointsData, int ImagePointCount, float *cudaImageData, int2 cudaImageSize, int2 featureU, int2 featureV, double *cudaFeatureValues) {

	// write data to global memory
    const unsigned int tid = threadIdx.x+(blockIdx.x*threadPerBlock);

	if (tid<ImagePointCount) {
		const int imageNum = cudaImagePointsData[tid].x;
		float* imagePtr = cudaImageData+(imageNum*cudaImageSize.x*cudaImageSize.y);
		int x = cudaImagePointsPos[tid].x; 
		int y = cudaImagePointsPos[tid].y; 

		//Calculate the vector position
		float dix = 1.0/(imagePtr[(y*cudaImageSize.x)+x] *0.005);
		float uxd = featureU.x*dix, uyd = featureU.y*dix;
		float vxd = featureV.x*dix, vyd = featureV.y*dix;
		float uX = x + uxd, uY = y + uyd;
		float vX = x + vxd, vY = y + vyd;

		//Check that the positions are within bounds
		float diU, diV;
		if (uX<0 || uY<0 || uX>=cudaImageSize.x || uY>=cudaImageSize.y) {
			diU = 10000;
		} else {
			diU = imagePtr[(int(uY)*cudaImageSize.x)+int(uX)];
		}
		if (vX<0 || vY<0 || vX>=cudaImageSize.x || vY>=cudaImageSize.y) {
			diV = 10000;
		} else {
			diV = imagePtr[(int(vY)*cudaImageSize.x)+int(vX)];
		}

		//If they fall on the background, set them to a high number
		if (diU==0.0) diU = 10000;
		if (diV==0.0) diV = 10000;

		//Return the value
		cudaFeatureValues[tid] = diU-diV;
	}
}

__global__ void
kernel_split_all_image_points(int threadPerBlock, int2 *cudaImagePointsPos, int2 *cudaImagePointsData, int ImagePointCount, float *cudaImageData, int2 cudaImageSize, int2* featureU, int2* featureV, double* featureT, int featureCount, bool *cudaSplitLeft) {

	// write data to global memory
    const unsigned int tid = threadIdx.x+(blockIdx.x*threadPerBlock);
	const unsigned int fid = blockIdx.y;

	if (tid<ImagePointCount) {
		const int imageNum = cudaImagePointsData[tid].x;
		float* imagePtr = cudaImageData+(imageNum*cudaImageSize.x*cudaImageSize.y);
		int x = cudaImagePointsPos[tid].x; 
		int y = cudaImagePointsPos[tid].y; 

		//Calculate the vector position
		float dix = 1.0/(imagePtr[(y*cudaImageSize.x)+x] *0.005);
		float uxd = featureU[fid].x*dix, uyd = featureU[fid].y*dix;
		float vxd = featureV[fid].x*dix, vyd = featureV[fid].y*dix;
		float uX = x + uxd, uY = y + uyd;
		float vX = x + vxd, vY = y + vyd;

		//Check that the positions are within bounds
		float diU, diV;
		if (uX<0 || uY<0 || uX>=cudaImageSize.x || uY>=cudaImageSize.y) {
			diU = 10000;
		} else {
			diU = imagePtr[(int(uY)*cudaImageSize.x)+int(uX)];
		}
		if (vX<0 || vY<0 || vX>=cudaImageSize.x || vY>=cudaImageSize.y) {
			diV = 10000;
		} else {
			diV = imagePtr[(int(vY)*cudaImageSize.x)+int(vX)];
		}

		//If they fall on the background, set them to a high number
		if (diU==0.0) diU = 10000;
		if (diV==0.0) diV = 10000;

		double val = diU-diV;
		if (val<featureT[fid]) {
			cudaSplitLeft[(fid*ImagePointCount)+tid] = true;
		} else {
			cudaSplitLeft[(fid*ImagePointCount)+tid] = false;
		}
	}
}

__global__ void
kernel_split_image_points(int threadPerBlock, int2 *cudaImagePointsPos, int2 *cudaImagePointsData, int ImagePointCount, float *cudaImageData, int2 cudaImageSize, int2 featureU, int2 featureV, double featureT, bool *cudaSplitLeft) {

	// write data to global memory
    const unsigned int tid = threadIdx.x+(blockIdx.x*threadPerBlock);

	if (tid<ImagePointCount) {
		const int imageNum = cudaImagePointsData[tid].x;
		float* imagePtr = cudaImageData+(imageNum*cudaImageSize.x*cudaImageSize.y);
		int x = cudaImagePointsPos[tid].x; 
		int y = cudaImagePointsPos[tid].y; 

		//Calculate the vector position
		float dix = 1.0/(imagePtr[(y*cudaImageSize.x)+x] *0.005);
		float uxd = featureU.x*dix, uyd = featureU.y*dix;
		float vxd = featureV.x*dix, vyd = featureV.y*dix;
		float uX = x + uxd, uY = y + uyd;
		float vX = x + vxd, vY = y + vyd;

		//Check that the positions are within bounds
		float diU, diV;
		if (uX<0 || uY<0 || uX>=cudaImageSize.x || uY>=cudaImageSize.y) {
			diU = 10000;
		} else {
			diU = imagePtr[(int(uY)*cudaImageSize.x)+int(uX)];
		}
		if (vX<0 || vY<0 || vX>=cudaImageSize.x || vY>=cudaImageSize.y) {
			diV = 10000;
		} else {
			diV = imagePtr[(int(vY)*cudaImageSize.x)+int(vX)];
		}

		//If they fall on the background, set them to a high number
		if (diU==0.0) diU = 10000;
		if (diV==0.0) diV = 10000;

		double val = diU-diV;
		if (val<featureT) {
			cudaSplitLeft[tid] = true;
		} else {
			cudaSplitLeft[tid] = false;
		}
	}
}

__global__ void 
kernel_calculate_class_counts(int threadPerBlock, int2* cudaImagePointsData, int cudaImagePointCount, bool* cudaSplitLeft, int classCount, int FeatureCount, int3* cudaClassCounts) {
	// Get thread ID
    const unsigned int tid = threadIdx.x+(blockIdx.x*threadPerBlock);
	
	//Check if it's value
	if (tid<FeatureCount) {
		//Calculate the split pointer position
		bool* splitPtr = cudaSplitLeft+(tid*cudaImagePointCount);

		//Loop through all the points
		for (int i=0; i<cudaImagePointCount; i++) {
			//Calculate the class
			int classIdx = cudaImagePointsData[i].y +(tid*(classCount+1));
			//Increment the before counter
			cudaClassCounts[classIdx].x++;
			//Increment the appropriate split counter
			if (splitPtr[i]) {
				cudaClassCounts[classIdx].y++;
			} else {
				cudaClassCounts[classIdx].z++;
			}
		}
	}
}


__global__ void
kernel_calculate_feature_gain(int threadPerBlock, int3* cudaClassCounts, int classCount, int FeatureCount, double* cudaFeatureGain) {

	const unsigned int tid = threadIdx.x+(blockIdx.x*threadPerBlock);

	if (tid<FeatureCount) {
		int3* classCountPtr = cudaClassCounts+(tid*(classCount+1));

		//Calculate total number of classes
		int nNodeClasses=0; int nLeftClasses=0; int nRightClasses=0; 
		for (int i=0; i<=classCount; i++) {
			nNodeClasses+=classCountPtr[i].x;
			nLeftClasses+=classCountPtr[i].y;
			nRightClasses+=classCountPtr[i].z;
		}

		double HC=0, HLC=0, HRC=0;
		double LC = nNodeClasses, LLC = nLeftClasses, LRC = nRightClasses;

		//Calculate Entropy
		if (nNodeClasses>0) {
			for (int i=0; i<=classCount; i++) {
				double countNormal = (double)(classCountPtr[i].x)/LC;
				if (countNormal!=0)
					HC += -(countNormal * log2f(countNormal));
			}
		}

		//Calculate Entropy
		if (nLeftClasses>0) {
			for (int i=0; i<=classCount; i++) {
				double countNormal = (double)(classCountPtr[i].y)/LLC;
				if (countNormal!=0)
					HLC += -(countNormal * log2f(countNormal));
			}
		}

		//Calculate Entropy
		if (nRightClasses>0) {
			for (int i=0; i<=classCount; i++) {
				double countNormal = (double)(classCountPtr[i].z)/LRC;
				if (countNormal!=0)
					HRC += -(countNormal * log2f(countNormal));
			}
		}

		cudaFeatureGain[tid] = HC - (((LLC/LC)*HLC)+((LRC/LC)*HRC));
	}
}



#endif // #ifndef _CPP_INTEGRATION_KERNEL_H_

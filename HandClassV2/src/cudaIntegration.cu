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
 * Host part of the device code.
 * Compiled with Cuda compiler.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "sdkHelper.h"  // helper for shared that are common to CUDA SDK samples
//#include <shrQATest.h>  // This is for automated testing output (--qatest)

// includes, kernels
#include "cudaKernel.cu"

//
////////////////////////////////////////////////////////////////////////////////
// declaration, forward


////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( cudaError err, const char *file, const int line )
    {
        if( cudaSuccess != err) {
		    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // This will output the proper error string when calling cudaGetLastError
    #define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

    inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // General GPU Device CUDA Initialization
    int gpuDeviceInit(int devID)
    {
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
            exit(-1);
        }
        if (devID < 0) 
            devID = 0;
        if (devID > deviceCount-1) {
            fprintf(stderr, "\n");
            fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
            fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
            fprintf(stderr, "\n");
            return -devID;
        }

        cudaDeviceProp deviceProp;
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        if (deviceProp.major < 1) {
            fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
            exit(-1);                                                  \
        }

        checkCudaErrors( cudaSetDevice(devID) );
        printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
        return devID;
    }

    // This function returns the best GPU (with maximum GFLOPS)
    int gpuGetMaxGflopsDeviceId()
    {
	    int current_device   = 0, sm_per_multiproc = 0;
	    int max_compute_perf = 0, max_perf_device  = 0;
	    int device_count     = 0, best_SM_arch     = 0;
	    cudaDeviceProp deviceProp;

	    cudaGetDeviceCount( &device_count );
	    // Find the best major SM Architecture GPU device
	    while ( current_device < device_count ) {
		    cudaGetDeviceProperties( &deviceProp, current_device );
		    if (deviceProp.major > 0 && deviceProp.major < 9999) {
			    best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		    }
		    current_device++;
	    }

        // Find the best CUDA capable GPU device
        current_device = 0;
        while( current_device < device_count ) {
           cudaGetDeviceProperties( &deviceProp, current_device );
           if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
               sm_per_multiproc = 1;
		   } else {
               sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
           }

           int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
           if( compute_perf  > max_compute_perf ) {
               // If we find GPU with SM major > 2, search only these
               if ( best_SM_arch > 2 ) {
                   // If our device==dest_SM_arch, choose this, or else pass
                   if (deviceProp.major == best_SM_arch) {	
                       max_compute_perf  = compute_perf;
                       max_perf_device   = current_device;
                   }
               } else {
                   max_compute_perf  = compute_perf;
                   max_perf_device   = current_device;
               }
           }
           ++current_device;
	    }
	    return max_perf_device;
    }

    // Initialization code to find the best CUDA Device
    int findCudaDevice(int argc, const char **argv)
    {
        cudaDeviceProp deviceProp;
        int devID = 0;
        // If the command-line has a device number specified, use it
        if (checkCmdLineFlag(argc, argv, "device")) {
            devID = getCmdLineArgumentInt(argc, argv, "device=");
            if (devID < 0) {
                printf("Invalid command line parameters\n");
                exit(-1);
            } else {
                devID = gpuDeviceInit(devID);
                if (devID < 0) {
                   printf("exiting...\n");
//                   shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                   exit(-1);
                }
            }
        } else {
            // Otherwise pick the device with highest Gflops/s
            devID = gpuGetMaxGflopsDeviceId();
            checkCudaErrors( cudaSetDevice( devID ) );
            checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
            printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
        }
        return devID;
    }
// end of CUDA Helper Functions

////////////////////////////////////////////////////////////////////////////////
//! Entry point for Cuda functionality on host side
//! @param argc  command line argument count
//! @param argv  command line arguments
//! @param data  data to process on the device
//! @param len   len of \a data
////////////////////////////////////////////////////////////////////////////////

extern "C" 
void cudaInit(const int argc, const char** argv) {
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, argv);
}


float *cudaImageData; int2 cudaImageSize;

extern "C" 
void cudaLoadImageData(float *vImageData, int imWidth, int imHeight, int imCount) {
    int memSize = imWidth*imHeight*imCount*sizeof(float);

	checkCudaErrors(cudaMalloc((void**) &cudaImageData, memSize));
	checkCudaErrors(cudaMemcpy(cudaImageData, vImageData, memSize, cudaMemcpyHostToDevice));

	cudaImageSize.x = imWidth; cudaImageSize.y = imHeight;
}

extern "C" 
void cudaFreeImageData() {
    checkCudaErrors(cudaFree(cudaImageData));
}

//In ImagePointData, x is the frame number, y is the class
int2* cudaImagePointsPos; int2* cudaImagePointsData; int cudaImagePointCount;

extern "C"
void cudaStoreImagePoints(int2 *ImagePointsPos, int2 *ImagePointsData, int ImagePointCount) {
    
	// allocate device memory
    cudaImagePointCount = ImagePointCount;
	int ipMemSize = cudaImagePointCount*sizeof(int2);
    checkCudaErrors(cudaMalloc((void**) &cudaImagePointsPos, ipMemSize));
    checkCudaErrors(cudaMemcpy(cudaImagePointsPos, ImagePointsPos, ipMemSize, cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMalloc((void**) &cudaImagePointsData, ipMemSize));
    checkCudaErrors(cudaMemcpy(cudaImagePointsData, ImagePointsData, ipMemSize, cudaMemcpyHostToDevice) );
}

extern "C"
void cudaFreeImagePoints() {
    checkCudaErrors(cudaFree(cudaImagePointsPos));
	checkCudaErrors(cudaFree(cudaImagePointsData));
}


extern "C" 
void cudaGetFeatureValues(double *FeatureValues, int2 featureU, int2 featureV) {
	
	// allocate return memory
    double* cudaFeatureValues; int fvMemSize = cudaImagePointCount*sizeof(double);
    checkCudaErrors(cudaMalloc((void**) &cudaFeatureValues, fvMemSize));

    // setup execution parameters
	int threadPerBlock = 512;
	int gridCount = ceil(double(cudaImagePointCount)/double(threadPerBlock));
	dim3 grid(gridCount);
    dim3 threads(threadPerBlock);

	// execute the kernel
    kernel_get_feature_values<<< grid, threads >>>(threadPerBlock, cudaImagePointsPos, cudaImagePointsData, cudaImagePointCount, cudaImageData, cudaImageSize, featureU, featureV, cudaFeatureValues);

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // copy results from device to host
    checkCudaErrors(cudaMemcpy(FeatureValues, cudaFeatureValues, fvMemSize, cudaMemcpyDeviceToHost));

    // cleanup memory
    checkCudaErrors(cudaFree(cudaFeatureValues));

    return;
}

extern "C" 
void cudaCalculateFeatureGain(int2* featureU, int2* featureV, double *featureT, int featureCount, int classCount, double *FeatureGain) {

	//First Calculate how each image point branches for each feature
	//Return a huge boolean array saying whether each imagepoint/feature
	//Splits left (true) or right (false).
	bool* cudaSplitLeft;
	{
		//Allocate Device Memory and Copy U,V,T
		int2* cudafeatureU, *cudafeatureV; int fMemSize = featureCount*sizeof(int2);
		checkCudaErrors(cudaMalloc((void**) &cudafeatureU, fMemSize));
		checkCudaErrors(cudaMalloc((void**) &cudafeatureV, fMemSize));
		checkCudaErrors(cudaMemcpy(cudafeatureU, featureU, fMemSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(cudafeatureV, featureV, fMemSize, cudaMemcpyHostToDevice));
		double* cudafeatureT; int ftMemSize = featureCount*sizeof(double);
		checkCudaErrors(cudaMalloc((void**) &cudafeatureT, ftMemSize));
		checkCudaErrors(cudaMemcpy(cudafeatureT, featureT, ftMemSize, cudaMemcpyHostToDevice));

		//Allocate Device Memory for Split Left Boolean
		int slMemSize = featureCount*cudaImagePointCount*sizeof(bool);
		checkCudaErrors(cudaMalloc((void**) &cudaSplitLeft, slMemSize));

		// Setup Kernel Parameters, threads*grid.x is for image points
		// grid.y is for the features
		int threadPerBlock = 512;
 		int gridCount = ceil(double(cudaImagePointCount)/double(threadPerBlock));
		dim3 grid(gridCount, featureCount);
		dim3 threads(threadPerBlock);

		// Execute the kernel
		kernel_split_all_image_points<<< grid, threads >>>
			(threadPerBlock, cudaImagePointsPos, cudaImagePointsData, cudaImagePointCount, cudaImageData, cudaImageSize, cudafeatureU, cudafeatureV, cudafeatureT, featureCount, cudaSplitLeft);

		// Check if kernel execution generated and error
		getLastCudaError("Kernel execution failed");

		// Cleanup memory
		checkCudaErrors(cudaFree(cudafeatureU));
		checkCudaErrors(cudaFree(cudafeatureV));
		checkCudaErrors(cudaFree(cudafeatureT));
	}

	//Next, for each feature calculate the number of each class before the split
	//and the number which split left, and right.
	int3* cudaClassCounts; 
	{
		//Allocate memory for the class counts
		int ccMemSize = featureCount*(classCount+1)*sizeof(int3);
		checkCudaErrors(cudaMalloc((void**) &cudaClassCounts, ccMemSize));
		checkCudaErrors(cudaMemset(cudaClassCounts, 0, ccMemSize));

 		// Setup Kernel Parameters, threads*grid.x is for feature counts
		int threadPerBlock = 512;
		int gridCount = ceil(double(featureCount)/double(threadPerBlock));
		dim3 grid(gridCount);
		dim3 threads(threadPerBlock);

		// Execute the kernel
		kernel_calculate_class_counts<<< grid, threads >>>
			(threadPerBlock, cudaImagePointsData, cudaImagePointCount, cudaSplitLeft, classCount, featureCount, cudaClassCounts);

		// Check if kernel execution generated and error
		getLastCudaError("Kernel execution failed");


	}

	// Cleanup the cudaSplitLeft boolean memory
	checkCudaErrors(cudaFree(cudaSplitLeft));

	//Finally, calculate the information gain for each feature. 
	//This is stored in an array of doubles
	double* cudaFeatureGain; 
	{
		//Allocate memory for the information gain
		int fgMemSize = featureCount*sizeof(double);
		checkCudaErrors(cudaMalloc((void**) &cudaFeatureGain, fgMemSize));
	
 		// Setup Kernel Parameters, threads*grid.x is for feature counts
		int threadPerBlock = 512;
		int gridCount = ceil(double(featureCount)/double(threadPerBlock));
		dim3 grid(gridCount);
		dim3 threads(threadPerBlock);

		// Execute the kernel
		kernel_calculate_feature_gain<<<grid, threads>>>
			(threadPerBlock, cudaClassCounts, classCount, featureCount, cudaFeatureGain);

		// check if kernel execution generated and error
		getLastCudaError("Kernel execution failed");
	}

	// Cleanup the cudaClassCounts int3 memory
	checkCudaErrors(cudaFree(cudaClassCounts));

    // Copy results from device to host
    checkCudaErrors(cudaMemcpy(FeatureGain, cudaFeatureGain, featureCount*sizeof(double), cudaMemcpyDeviceToHost));

    // cleanup memory
    checkCudaErrors(cudaFree(cudaFeatureGain));

    return;
}

extern "C" 
void cudaSplitImagePoints(bool *splitLeft, int2 featureU, int2 featureV, double featureT) {

	// allocate return memory
    bool* cudaSplitLeft; int slMemSize = cudaImagePointCount*sizeof(bool);
    checkCudaErrors(cudaMalloc((void**) &cudaSplitLeft, slMemSize));

    // setup execution parameters
	int threadPerBlock = 512;
	int gridCount = ceil(double(cudaImagePointCount)/double(threadPerBlock));
    dim3 grid(gridCount);
    dim3 threads(threadPerBlock);

	// execute the kernel
    kernel_split_image_points<<< grid, threads >>>
		(threadPerBlock, cudaImagePointsPos, cudaImagePointsData, cudaImagePointCount, cudaImageData, cudaImageSize, featureU, featureV, featureT, cudaSplitLeft);

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // copy results from device to host
    checkCudaErrors(cudaMemcpy(splitLeft, cudaSplitLeft, slMemSize, cudaMemcpyDeviceToHost));

    // cleanup memory
    checkCudaErrors(cudaFree(cudaSplitLeft));

    return;
}
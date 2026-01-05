#if 0
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

#include <ImfArray.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>

using namespace Imf;
using namespace Imath;
using namespace std;

void myMouse(int _event, int x, int y, int flags, void* param) {
	printf("%f\r", CV_IMAGE_ELEM((IplImage*)param, float, y, x));
}

vector<IplImage*> GetChannels(const char * filename, int& width, int& height);

void main() {

	char filename[50];
	for (int i=0; i<=60; i++) {
		sprintf(filename, "translation2/translation3_ZDepth%04d.exr", i);

		int width, height;
		vector<IplImage*> Channels = GetChannels(filename, width, height);


	/*	IplImage *mask = cvCreateImage(cvGetSize(Channels.at(0)), IPL_DEPTH_8U, 1); cvSetZero(mask);
		for (int y=0; y<height; y++) {
			for (int x=0; x<width; x++) {
				if (CV_IMAGE_ELEM(Channels.at(0), float, y, x)!=0) 
					CV_IMAGE_ELEM(mask, unsigned char, y, x)=255;
			}
		}

		cvShowImage("mask", mask);

		IplImage *display = cvCreateImage(cvGetSize(Channels.at(0)), IPL_DEPTH_32F, 1); cvSet(display, cvScalar(1.0));
		//cvSubS(Channels.at(0), cvScalar(
		cvSub(display, Channels.at(0), display);
		cvShowImage("depthInv", display);

		double min, max;
		cvMinMaxLoc(Channels.at(0), &min, &max,0, 0, mask);
		printf("%f, %f\n", min, max);*/
		cvShowImage(filename, Channels.at(0));

		cvSetMouseCallback(filename, myMouse, Channels.at(0));

	}
	cvWaitKey();


}

vector<IplImage*> GetChannels(const char * filename, int& width, int& height) {
		
	InputFile file(filename);

	Box2i dw = file.header().dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;

	FrameBuffer frameBuffer;

	std::vector<IplImage *> Channels;

	ChannelList c = file.header().channels();
	for (ChannelList::ConstIterator i = c.begin(); i!=c.end(); i++) {
		Channel c = i.channel();

		if (c.type == 0) {
			// unsigned int (32 bit)
			IplImage *channel = cvCreateImage(cvSize(width, height), IPL_DEPTH_32S, 1);
			frameBuffer.insert (i.name(), Slice (c.type, channel->imageData, sizeof (long),	sizeof (long) * width, 1, 1, 0.0)); 
			Channels.push_back(channel);

		} else if (c.type == 1) {
			Array2D<short> pBuffer;			// half (16 bit floating point)
			pBuffer.resizeErase (height, width);
			frameBuffer.insert (i.name(), Slice (c.type, (char *) (&pBuffer[0][0] - dw.min.x - dw.min.y * width), 
			sizeof (pBuffer[0][0]) * 1,	sizeof (pBuffer[0][0]) * width, 1, 1, 0.0)); 


		} else if (c.type == 2) {
			// float (32 bit floating point)
			IplImage *channel = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
			frameBuffer.insert (i.name(), Slice (c.type, channel->imageData, sizeof (float), sizeof (float) * width, 1, 1, 0.0)); 
			Channels.push_back(channel);
		}



	}

	file.setFrameBuffer (frameBuffer);
	file.readPixels (dw.min.y, dw.max.y);

	return Channels;
}
#endif
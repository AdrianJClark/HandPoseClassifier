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
	int width, height;
	vector<IplImage *> channels = GetChannels("translation2_ZDepth.exr", width, height);

	for (int i=0; i<channels.size(); i++) {
		char title[50]; sprintf(title, "Channel %d", i);
		IplImage *displayImage = cvCreateImage(cvGetSize(channels.at(0)), IPL_DEPTH_8U, 1);

		double min, max;
		cvMinMaxLoc(channels.at(0), &min, &max);
		cvConvertScale(channels.at(0), displayImage, 255.0/(max-min));

		cvShowImage(title, displayImage);
		cvSetMouseCallback(title, myMouse, channels.at(i));
		cvReleaseImage(&displayImage);
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
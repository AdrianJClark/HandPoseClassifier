#ifndef SETTINGS_H
#define SETTINGS_H

#include <tinyxml.h>

int frameBegin, frameEnd;
std::string colorInPath, depthInPath;
std::string colorTable;

std::string colorOutPath, classOutPath, depthOutPath;

void loadSettings(const char* filename) {

	TiXmlDocument doc(filename);
	if (!doc.LoadFile()) { printf ("Could not find settings file. Exiting\n"); return; }

	colorInPath = doc.FirstChildElement("ColorInFileString")->GetText();
	depthInPath = doc.FirstChildElement("DepthInFileString")->GetText();
	colorTable = doc.FirstChildElement("ColorTableFile")->GetText();
	colorOutPath = doc.FirstChildElement("ColorOutFileString")->GetText();
	classOutPath = doc.FirstChildElement("ClassOutFileString")->GetText();
	depthOutPath = doc.FirstChildElement("DepthOutFileString")->GetText();
	frameBegin = atoi(doc.FirstChildElement("FrameBegin")->GetText());
	frameEnd = atoi(doc.FirstChildElement("FrameEnd")->GetText());


}

#endif
#ifndef SETTINGS_H
#define SETTINGS_H

#include <tinyxml.h>

int frameBegin, frameEnd;
int maxTreeDepth;
int classCount;

std::string depthPath, classPath;
std::string featuresFileName;
std::string treeFileName, intTreeFileName;

void loadSettings(const char* filename) {

	TiXmlDocument doc(filename);
	if (!doc.LoadFile()) { printf ("Could not find settings file. Exiting\n"); return; }

	featuresFileName = doc.FirstChildElement("FeatureFile")->GetText();
	depthPath = doc.FirstChildElement("DepthFileString")->GetText();
	classPath = doc.FirstChildElement("ClassFileString")->GetText();
	treeFileName = doc.FirstChildElement("OutputTreeName")->GetText();
	intTreeFileName = doc.FirstChildElement("IntermediateTreeName")->GetText();
	frameBegin = atoi(doc.FirstChildElement("FrameStart")->GetText());
	frameEnd = atoi(doc.FirstChildElement("FrameEnd")->GetText());
	maxTreeDepth = atoi(doc.FirstChildElement("MaxTreeDepth")->GetText());
	classCount = atoi(doc.FirstChildElement("NumberOfClasses")->GetText());

	if (maxTreeDepth==-1) maxTreeDepth = INT_MAX;

}

#endif
#ifndef COLORCONVERT_H
#define COLORCONVERT_H

#include <map>
struct ClassColor {
	int _class;
	int r, g, b;
};

std::map<int, ClassColor> ColorMap;

void loadColorTable(const char* filename) {
	FILE *f = fopen(filename, "rb");
	ClassColor c;
	while (fscanf(f, "%d=%d,%d,%d\r\n", &c._class, &c.r, &c.g, &c.b)==4) {
		ColorMap[c._class] = c;
	}
}

#endif
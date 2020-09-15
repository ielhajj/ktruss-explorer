
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#ifndef _INFO_H_
#define _INFO_H_

#include "config.h"

#define NOT_AFFECTED        0
#define DIRECTLY_AFFECTED   1
#define INDIRECTLY_AFFECTED 2

struct Info {
    unsigned int* changed;
    unsigned int* numTriangles;
    unsigned int* nodeAffected;
    unsigned int* edgeAffected;
};

Info createInfo(Config config);
Info createInfoOnDevice(Config config);

void initInfo(Info info, Config config);
void initInfoOnDevice(Info info, Config config);

void clearIterInfo(Info info, Config config);
void clearIterInfoOnDevice(Info info, Config config);

void freeInfo(Info info);
void freeInfoOnDevice(Info info);

#endif


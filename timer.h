
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>
#include <sys/time.h>

enum PrintColor { NONE, GREEN, CYAN };

struct Timer {
    unsigned int isActive;
    struct timeval startTime;
    struct timeval endTime;
};

static Timer initTimer(unsigned int isActive) {
    Timer timer;
    timer.isActive = isActive;
    return timer;
}

static void startTimer(Timer* timer) {
    if(timer->isActive) {
        gettimeofday(&(timer->startTime), NULL);
    }
}

static void printAndStart(Timer* timer, const char* s) {
    if(timer->isActive) {
        printf("%s", s);
        startTimer(timer);
    }
}

static void stopTimer(Timer* timer) {
    if(timer->isActive) {
        gettimeofday(&(timer->endTime), NULL);
    }
}

static void printElapsedTime(Timer* timer, const char* s = NULL, enum PrintColor color = NONE) {
    if(timer->isActive) {
        float t = ((float) ((timer->endTime.tv_sec - timer->startTime.tv_sec) \
                        + (timer->endTime.tv_usec - timer->startTime.tv_usec)/1.0e6));
        switch(color) {
            case GREEN: printf("\033[0;32m"); break;
            case CYAN : printf("\033[0;36m"); break;
        }
        if(s == NULL) {
            printf("%f ms\n", t*1e3);
        } else {
            printf("%s: %f ms\n", s, t*1e3);
        }
        if(color != NONE) {
            printf("\033[0m");
        }
    }
}

static void stopAndPrintElapsed(Timer* timer, const char* s = NULL, enum PrintColor color = NONE) {
    if(timer->isActive) {
        stopTimer(timer);
        printElapsedTime(timer, s, color);
    }
}

static void syncStopAndPrintElapsed(Timer* timer, const char* s = NULL, enum PrintColor color = NONE) {
    if(timer->isActive) {
        cudaDeviceSynchronize();
        stopAndPrintElapsed(timer, s, color);
    }
}

#endif


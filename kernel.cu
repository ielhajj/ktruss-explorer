
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "graph.h"
#include "kernel.h"
#include "timer.h"

void ktruss(COOGraph* graph, COOGraph* truss, Config config) {

    // Relabel vertices
    unsigned int* new2old = NULL;
    if(config.directedness == DIRECTED_BY_DEGREE) {
        Timer timer = initTimer(config.verbosity >= 1);
        startTimer(&timer);
        new2old = (unsigned int*) malloc(graph->numNodes*sizeof(unsigned int));
        // TODO: Implement relabeling on GPU
        sortByDegree(graph, new2old);
        stopAndPrintElapsed(&timer, "    Relabel vertices (not optimized)");
    }

    // K-truss
    switch(config.processor) {
        case CPU: Host::ktruss(graph, truss, config);   break;
        case GPU: Device::ktruss(graph, truss, config); break;
        default: printf("Version not supported!\n"); exit(0);
    }

    // Restore original vertex labels
    if(config.directedness == DIRECTED_BY_DEGREE) {
        Timer timer = initTimer(config.verbosity >= 1);
        startTimer(&timer);
        // TODO: Implement relabeling on GPU
        unsort(truss, new2old);
        free(new2old);
        stopAndPrintElapsed(&timer, "    Restore original vertex labels (not optimized)");
    }

}


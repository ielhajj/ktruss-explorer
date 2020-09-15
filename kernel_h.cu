
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "graph.h"
#include "kernel.h"
#include "timer.h"

namespace Host {

void ktruss(COOGraph* graph, COOGraph* truss, Config config) {

    Timer timer = initTimer(config.verbosity >= 1);

    // Allocate intermediate data structures
    startTimer(&timer);
    Info info = createInfo(config);
    syncStopAndPrintElapsed(&timer, "    Allocate intermediate data structures");

    // Convert undirected COO to directed COO
    COOGraph* gdir = NULL;
    if(isDirected(config.directedness)) {
        startTimer(&timer);
        gdir = createEmptyCOO(config.numNodes, config.numEdges/2);
        undirected2directedCOO(graph, gdir);
        stopAndPrintElapsed(&timer, "    Convert undirected COO to directed COO");
    }

    // K-truss
    switch(config.directedness) {
        case UNDIRECTED:
            switch(config.inputFormat) {
                case COOCSR:        Undirected::COOCSRInput::ktruss(graph, truss, info, config);      break;
                case TILED_COOCSR:  Undirected::TiledCOOCSRInput::ktruss(graph, truss, info, config); break;
                default: printf("Version not supported!\n"); exit(0);
            }
            break;
        case DIRECTED_BY_INDEX:
        case DIRECTED_BY_DEGREE:
            switch(config.inputFormat) {
                case COOCSR:        Directed::COOCSRInput::ktruss(gdir, truss, info, config);         break;
                case TILED_COOCSR:  Directed::TiledCOOCSRInput::ktruss(gdir, truss, info, config);    break;
                default: printf("Version not supported!\n"); exit(0);
            }
            break;
        default: printf("Version not supported!\n"); exit(0);
    }

    // Convert directed COO to undirected COO
    if(isDirected(config.directedness)) {
        startTimer(&timer);
        directed2undirectedCOO(truss);
        stopAndPrintElapsed(&timer, "    Convert directed COO to undirected COO");
    }

    // Deallocate directed COO
    if(isDirected(config.directedness)) {
        startTimer(&timer);
        freeCOOGraph(gdir);
        stopAndPrintElapsed(&timer, "    Deallocate directed COO");
    }

    // Deallocate intermediate data structures
    startTimer(&timer);
    freeInfo(info);
    syncStopAndPrintElapsed(&timer, "    Deallocate intermediate data structures");

}

}


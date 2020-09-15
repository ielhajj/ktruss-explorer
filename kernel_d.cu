
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "graph.h"
#include "kernel.h"
#include "timer.h"

namespace Device {

void ktruss(COOGraph* graph, COOGraph* truss, Config config) {

    Timer timer = initTimer(config.verbosity >= 1);

    // Copy COO graph to GPU
    startTimer(&timer);
    COOGraph* graph_d = createEmptyCOOOnDevice(config.numNodes, config.numEdges);
    COOGraph* truss_d = createEmptyCOOOnDevice(config.numNodes, config.numEdges);
    copyCOOToDevice(graph, graph_d);
    syncStopAndPrintElapsed(&timer, "    Copy COO graph to GPU");

    // Allocate intermediate data structures
    startTimer(&timer);
    Info info = createInfoOnDevice(config);
    syncStopAndPrintElapsed(&timer, "    Allocate intermediate data structures");

    // Convert undirected COO to directed COO
    COOGraph* gdir_d = NULL;
    if(isDirected(config.directedness)) {
        startTimer(&timer);
        gdir_d = createEmptyCOOOnDevice(config.numNodes, config.numEdges/2);
        undirected2directedCOOOnDevice(graph_d, gdir_d);
        syncStopAndPrintElapsed(&timer, "    Convert undirected COO to directed COO");
    }

    // K-truss
    switch(config.directedness) {
        case UNDIRECTED:
            switch(config.inputFormat) {
                case COOCSR:        Undirected::COOCSRInput::ktruss(graph_d, truss_d, info, config);      break;
                case TILED_COOCSR:  Undirected::TiledCOOCSRInput::ktruss(graph_d, truss_d, info, config); break;
                default: printf("Version not supported!\n"); exit(0);
            }
            break;
        case DIRECTED_BY_INDEX:
        case DIRECTED_BY_DEGREE:
            switch(config.inputFormat) {
                case COOCSR:        Directed::COOCSRInput::ktruss(gdir_d, truss_d, info, config);         break;
                case TILED_COOCSR:  Directed::TiledCOOCSRInput::ktruss(gdir_d, truss_d, info, config);    break;
                default: printf("Version not supported!\n"); exit(0);
            }
            break;
        default: printf("Version not supported!\n"); exit(0);
    }

    // Convert directed COO to undirected COO
    if(isDirected(config.directedness)) {
        startTimer(&timer);
        directed2undirectedCOOOnDevice(truss_d);
        syncStopAndPrintElapsed(&timer, "    Convert directed COO to undirected COO");
    }

    // Deallocate directed COO
    if(isDirected(config.directedness)) {
        startTimer(&timer);
        freeCOOGraphOnDevice(gdir_d);
        syncStopAndPrintElapsed(&timer, "    Deallocate directed COO");
    }

    // Deallocate intermediate data structures
    startTimer(&timer);
    freeInfoOnDevice(info);
    syncStopAndPrintElapsed(&timer, "    Deallocate intermediate data structures");

    // Copy COO truss from GPU
    startTimer(&timer);
    copyCOOFromDevice(truss_d, truss);
    syncStopAndPrintElapsed(&timer, "    Copy COO truss from GPU");

    // Deallocate graph and truss from GPU
    startTimer(&timer);
    freeCOOGraphOnDevice(graph_d);
    freeCOOGraphOnDevice(truss_d);
    syncStopAndPrintElapsed(&timer, "    Deallocate COO graph and truss from GPU");

}

}



// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "config.h"
#include "graph.h"
#include "kernel.h"
#include "timer.h"
#include "verify.h"

#include <stdio.h>

int main(int argc, char** argv) {

    cudaDeviceSynchronize();

    Config config = parseArgs(argc, argv);
    Timer timer = initTimer(config.verbosity >= 1);

    // Allocate memory and initialize data
    printAndStart(&timer, "Setting up\n");
    COOGraph* graph = createCOOFromFile(config.graphFileName);
    config.numNodes = graph->numNodes;
    config.numEdges = graph->numEdges;
    COOGraph* truss = createEmptyCOO(config.numNodes, config.numEdges);
    stopAndPrintElapsed(&timer, "    Set up time");
    if(config.verbosity >= 1) printConfig(config);

    // Compute
    printAndStart(&timer, "Computing\n");
    ktruss(graph, truss, config);
    stopAndPrintElapsed(&timer, "    Total time");

    // Write result
    if(config.outFileName != NULL) {
        if(config.verbosity >= 1) printf("Writing result to file: %s\n", config.outFileName);
        writeCOOGraphToFile(truss, config.outFileName);
    }

    // Verify result
    printAndStart(&timer, "Verifying result\n");
    verify(truss, config);
    stopAndPrintElapsed(&timer, "    Verification time");

    // Free memory
    printAndStart(&timer, "Cleaning up\n");
    freeCOOGraph(graph);
    freeCOOGraph(truss);
    stopAndPrintElapsed(&timer, "    Clean up time");

    return 0;

}


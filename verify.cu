
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "graph.h"

#include <assert.h>
#include <stdio.h>

#include "verify.h"

void verify(COOGraph* graph, Config config) {

    // Read reference from file
    COOGraph* reference = createCOOFromFile(config.compareFileName);

    // Check edge count
    assert(graph->numEdges == reference->numEdges);

    // Convert graphs to COO/CSR
    COOCSRGraph* g = createEmptyCOOCSR(graph->numNodes, graph->numEdges);
    COOCSRGraph* gr = createEmptyCOOCSR(reference->numNodes, reference->numEdges);
    coo2coocsr(graph, g);
    coo2coocsr(reference, gr);

    // Compare COO/CSR graphs
    for(unsigned int e = 0; e < g->numEdges; ++e) {
        assert(g->srcIdx[e] == gr->srcIdx[e]);
        assert(g->dstIdx[e] == gr->dstIdx[e]);
    }

    // Free COO/CSR graphs
    freeCOOCSRGraph(g);
    freeCOOCSRGraph(gr);
    freeCOOGraph(reference);

    // Passed
    if(config.verbosity >= 1) printf("    PASSED\n");

}


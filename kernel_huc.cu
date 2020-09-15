
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "graph.h"
#include "kernel.h"
#include "timer.h"

namespace Host { namespace Undirected { namespace COOCSRInput {

template < unsigned int CHECK_IF_DELETED, unsigned int RECOUNT_ALL_EDGES >
void count_triangles(COOCSRGraph* g, unsigned int k, Info info) {
    for(unsigned int e = 0; e < g->numEdges; ++e) {
        if(RECOUNT_ALL_EDGES || info.edgeAffected[e] == DIRECTLY_AFFECTED) {
            unsigned int dst = g->dstIdx[e];
            if(!CHECK_IF_DELETED || dst != DELETED) {
                unsigned int src1 = g->srcIdx[e];
                unsigned int src2 = dst;
                unsigned int e1 = g->srcPtr[src1];
                unsigned int e2 = g->srcPtr[src2];
                unsigned int end1 = g->srcPtr[src1 + 1];
                unsigned int end2 = g->srcPtr[src2 + 1];
                if(end1 - e1 >= k - 2 && end2 - e2 >= k - 2) {
                    while(e1 < end1 && e2 < end2 && info.numTriangles[e] < k - 2) {
                        unsigned int dst1 = g->dstIdx[e1];
                        if(CHECK_IF_DELETED && dst1 == DELETED) {
                            ++e1;
                        } else {
                            unsigned int dst2 = g->dstIdx[e2];
                            if(CHECK_IF_DELETED && dst2 == DELETED) {
                                ++e2;
                            } else {
                                if(dst1 < dst2) {
                                    ++e1;
                                } else if(dst1 > dst2) {
                                    ++e2;
                                } else { // dst1 == dst2
                                    ++e1;
                                    ++e2;
                                    ++info.numTriangles[e];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template < unsigned int CHECK_IF_DELETED, unsigned int RECOUNT_ALL_EDGES >
void mark_deleted_edges(COOCSRGraph* g, unsigned int k, Info info) {
    for(unsigned int e = 0; e < g->numEdges; ++e) {
        if(RECOUNT_ALL_EDGES || info.edgeAffected[e] == DIRECTLY_AFFECTED) {
            unsigned int dst = g->dstIdx[e];
            if(!CHECK_IF_DELETED || dst != DELETED) {
                if(info.numTriangles[e] < k - 2) {
                    g->dstIdx[e] = DELETED;
                    *info.changed = 1;
                    if(!RECOUNT_ALL_EDGES) {
                        // If only affected edges are going to be recounted, mark which nodes are directly affected
                        unsigned int src = g->srcIdx[e];
                        info.nodeAffected[src] = DIRECTLY_AFFECTED;
                        info.nodeAffected[dst] = DIRECTLY_AFFECTED;
                    }
                }
            }
        }
    }
}

template < unsigned int CHECK_IF_DELETED >
void mark_directly_affected_edges(COOCSRGraph* g, Info info) {
    for(unsigned int e = 0; e < g->numEdges; ++e) {
        unsigned int edgeAffected_e = NOT_AFFECTED;
        unsigned int dst = g->dstIdx[e];
        if(!CHECK_IF_DELETED || dst != DELETED) {
            unsigned int src = g->srcIdx[e];
            if(info.nodeAffected[src] == DIRECTLY_AFFECTED || info.nodeAffected[dst] == DIRECTLY_AFFECTED) {
                edgeAffected_e = DIRECTLY_AFFECTED;
            }
        }
        info.edgeAffected[e] = edgeAffected_e;
    }
}

void ktruss(COOCSRGraph* g, Info info, Config config) {

    unsigned int k = config.k;
    unsigned int iter = 0;
    unsigned int graphHasDeletedEdges = 0;
    unsigned int changed;
    initInfo(info, config);
    do {

        if(config.verbosity >= 2) printf("        Iteration %u\n", iter);
        clearIterInfo(info, config);

        // Count triangles
        Timer iterTimer = initTimer(config.verbosity >= 2);
        startTimer(&iterTimer);
        if(graphHasDeletedEdges) {
            if(config.recount == ALL) {
                count_triangles<1,1>(g, k, info);
            } else { // config.recount == AFFECTED
                count_triangles<1,0>(g, k, info);
            }
        } else {
            if(config.recount == ALL) {
                count_triangles<0,1>(g, k, info);
            } else { // config.recount == AFFECTED
                count_triangles<0,0>(g, k, info);
            }
        }
        stopAndPrintElapsed(&iterTimer, "            Triangle counting time");

        // Mark deleted edges
        startTimer(&iterTimer);
        if(graphHasDeletedEdges) {
            if(config.recount == ALL) {
                mark_deleted_edges<1,1>(g, k, info);
            } else { // config.recount == AFFECTED
                mark_deleted_edges<1,0>(g, k, info);
            }
        } else {
            if(config.recount == ALL) {
                mark_deleted_edges<0,1>(g, k, info);
            } else { // config.recount == AFFECTED
                mark_deleted_edges<0,0>(g, k, info);
            }
        }
        stopAndPrintElapsed(&iterTimer, "            Mark deleted edges");

        // Check if the graph changed
        changed = *info.changed;
        if(changed) {

            // Remove deleted edges
            if(iter < config.numEdgeRemoveIter) {
                startTimer(&iterTimer);
                removeCOOCSRDeletedEdges(g);
                graphHasDeletedEdges = 0;
                stopAndPrintElapsed(&iterTimer, "            Remove deleted edges");
                if(config.verbosity >= 2) printf("                # edges remaining = %u\n", g->numEdges);
            } else {
                graphHasDeletedEdges = 1;
            }

            // If k=3, no need to recount
            if(k == 3) {
                break;
            }

            // Mark affected edges
            if(config.recount == AFFECTED) {
                startTimer(&iterTimer);
                if(graphHasDeletedEdges) {
                    mark_directly_affected_edges<1>(g, info);
                } else {
                    mark_directly_affected_edges<0>(g, info);
                }
                syncStopAndPrintElapsed(&iterTimer, "            Mark affected edges");
            }

        }

        ++iter;

    } while(changed);

    // Remove deleted edges
    if(graphHasDeletedEdges) {
        Timer timer = initTimer(config.verbosity >= 2);
        startTimer(&timer);
        removeCOOCSRDeletedEdges(g);
        stopAndPrintElapsed(&timer, "        Remove deleted edges");
    }

}

void ktruss(COOGraph* graph, COOGraph* truss, Info info, Config config) {

    Timer timer = initTimer(config.verbosity >= 1);

    // Convert COO to COOCSR
    startTimer(&timer);
    COOCSRGraph* g = createEmptyCOOCSR(config.numNodes, config.numEdges);
    coo2coocsr(graph, g);
    stopAndPrintElapsed(&timer, "    Convert undirected COO to COOCSR");

    // Runs
    for(unsigned int i = 0; i < config.numWarmupRuns + config.numTimedRuns; ++i) {

        // K-truss
        printAndStart(&timer, "    Performing K-truss\n");
        Timer ktrussTimer = initTimer(config.verbosity == 0 && i >= config.numWarmupRuns);
        startTimer(&ktrussTimer);
        ktruss(g, info, config);
        if(config.verbosity == 0 && i >= config.numWarmupRuns) printConfigAsCSV(config);
        stopAndPrintElapsed(&ktrussTimer);
        stopAndPrintElapsed(&timer, "        Total K-truss time", CYAN);

        // Restore graph
        if(i < config.numWarmupRuns + config.numTimedRuns - 1) {
            coo2coocsr(graph, g);
        }

    }

    // Convert COOCSR to COO
    startTimer(&timer);
    coocsr2coo(g, truss);
    stopAndPrintElapsed(&timer, "    Convert COO/CSR to COO");

    // Deallocate COOCSR
    startTimer(&timer);
    freeCOOCSRGraph(g);
    stopAndPrintElapsed(&timer, "    Deallocate COOCSR");

}

} } } // end namespace


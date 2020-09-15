
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "graph.h"
#include "kernel.h"
#include "timer.h"

namespace Host { namespace Directed { namespace TiledCOOCSRInput {

template < unsigned int CHECK_IF_DELETED, unsigned int RECOUNT_ALL_EDGES >
void count_triangles(TiledCOOCSRGraph* g, Info info) {
    unsigned int tileSize = g->tileSize;
    unsigned int tilesPerDim = g->tilesPerDim;
    for(unsigned int srcTile = 0; srcTile < tilesPerDim; ++srcTile) {
        for(unsigned int dstTile = 0; dstTile < tilesPerDim; ++dstTile) {
            unsigned int tileOffset = (srcTile*tilesPerDim + dstTile)*tileSize;
            unsigned int tileEdgesStart = g->tileSrcPtr[tileOffset];
            unsigned int tileEdgesEnd = g->tileSrcPtr[tileOffset + tileSize];
            for(unsigned int xTile = dstTile; xTile < tilesPerDim; ++xTile) {
                for(unsigned int e = tileEdgesStart; e < tileEdgesEnd; ++e) {
                    if(RECOUNT_ALL_EDGES || info.edgeAffected[e] == DIRECTLY_AFFECTED || info.edgeAffected[e] == INDIRECTLY_AFFECTED) {
                        unsigned int dst = g->dstIdx[e];
                        if(!CHECK_IF_DELETED || dst != DELETED) {
                            unsigned int src1 = g->srcIdx[e];
                            unsigned int src2 = dst;
                            unsigned int tileSrc1 = (srcTile*tilesPerDim + xTile)*tileSize + src1%tileSize;
                            unsigned int tileSrc2 = (dstTile*tilesPerDim + xTile)*tileSize + src2%tileSize;
                            unsigned int e1 = g->tileSrcPtr[tileSrc1];
                            unsigned int e2 = g->tileSrcPtr[tileSrc2];
                            unsigned int end1 = g->tileSrcPtr[tileSrc1 + 1];
                            unsigned int end2 = g->tileSrcPtr[tileSrc2 + 1];
                            while(e1 < end1 && e2 < end2) {
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
                                            ++info.numTriangles[e];
                                            ++info.numTriangles[e1];
                                            ++info.numTriangles[e2];
                                            ++e1;
                                            ++e2;
                                        }
                                    }
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
void mark_deleted_edges(TiledCOOCSRGraph* g, unsigned int k, Info info) {
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
void mark_directly_affected_edges(TiledCOOCSRGraph* g, Info info) {
    for(unsigned int e = 0; e < g->numEdges; ++e) {
        unsigned int edgeAffected_e = NOT_AFFECTED;
        unsigned int dst = g->dstIdx[e];
        if(!CHECK_IF_DELETED || dst != DELETED) {
            unsigned int src = g->srcIdx[e];
            if(info.nodeAffected[src] == DIRECTLY_AFFECTED || info.nodeAffected[dst] == DIRECTLY_AFFECTED) {
                edgeAffected_e = DIRECTLY_AFFECTED;
                if(info.nodeAffected[src] != DIRECTLY_AFFECTED) {
                    info.nodeAffected[src] = INDIRECTLY_AFFECTED;
                } else if(info.nodeAffected[dst] != DIRECTLY_AFFECTED) {
                    info.nodeAffected[dst] = INDIRECTLY_AFFECTED;
                }
            }
        }
        info.edgeAffected[e] = edgeAffected_e;
    }
}

template < unsigned int CHECK_IF_DELETED >
void mark_indirectly_affected_edges(TiledCOOCSRGraph* g, Info info) {
    for(unsigned int e = 0; e < g->numEdges; ++e) {
        unsigned int dst = g->dstIdx[e];
        if(!CHECK_IF_DELETED || dst != DELETED) {
            if(info.edgeAffected[e] != DIRECTLY_AFFECTED) {
                unsigned int src = g->srcIdx[e];
                if(info.nodeAffected[src] == INDIRECTLY_AFFECTED || info.nodeAffected[dst] == INDIRECTLY_AFFECTED) {
                    info.edgeAffected[e] = INDIRECTLY_AFFECTED;
                }
            }
        }
    }
}

void ktruss(TiledCOOCSRGraph* g, Info info, Config config) {

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
                count_triangles<1,1>(g, info);
            } else { // config.recount == AFFECTED
                count_triangles<1,0>(g, info);
            }
        } else {
            if(config.recount == ALL) {
                count_triangles<0,1>(g, info);
            } else { // config.recount == AFFECTED
                count_triangles<0,0>(g, info);
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
                removeTiledCOOCSRDeletedEdges(g);
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
                    mark_indirectly_affected_edges<1>(g, info);
                } else {
                    mark_directly_affected_edges<0>(g, info);
                    mark_indirectly_affected_edges<0>(g, info);
                }
                stopAndPrintElapsed(&iterTimer, "            Mark affected edges");
            }

        }

        ++iter;

    } while(changed);

    // Remove deleted edges
    if(graphHasDeletedEdges) {
        Timer timer = initTimer(config.verbosity >= 2);
        startTimer(&timer);
        removeTiledCOOCSRDeletedEdges(g);
        stopAndPrintElapsed(&timer, "        Remove deleted edges");
    }

}

void ktruss(COOGraph* gdir, COOGraph* truss, Info info, Config config) {

    Timer timer = initTimer(config.verbosity >= 1);

    // Convert COO to tiled COOCSR
    startTimer(&timer);
    TiledCOOCSRGraph* g = createEmptyTiledCOOCSR(config.numNodes, config.numTiles, config.numEdges/2);
    coo2tiledcoocsr(gdir, g);
    stopAndPrintElapsed(&timer, "    Convert directed COO to tiled COOCSR");

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
            coo2tiledcoocsr(gdir, g);
        }

    }

    // Convert tiled COOCSR to COO
    startTimer(&timer);
    tiledcoocsr2coo(g, truss);
    stopAndPrintElapsed(&timer, "    Convert tiled COOCSR to COO");

    // Deallocate tiled COOCSR
    startTimer(&timer);
    freeTiledCOOCSRGraph(g);
    stopAndPrintElapsed(&timer, "    Deallocate tiled COOCSR");

}

} } } // end namespace


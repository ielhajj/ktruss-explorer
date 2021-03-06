
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "graph.h"
#include "kernel.h"
#include "timer.h"

namespace Device { namespace Undirected { namespace TiledCOOCSRInput {

template < unsigned int CHECK_IF_DELETED, unsigned int RECOUNT_ALL_EDGES >
__global__ void count_triangles_kernel(TiledCOOCSRGraph* g, unsigned int k, Info info) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < g->numEdges) {
        if(RECOUNT_ALL_EDGES || info.edgeAffected[e] == DIRECTLY_AFFECTED) {
            unsigned int* srcIdx = g->srcIdx;
            unsigned int* dstIdx = g->dstIdx;
            unsigned int* tileSrcPtr = g->tileSrcPtr;
            unsigned int dst = dstIdx[e];
            if(!CHECK_IF_DELETED || dst != DELETED) {
                unsigned int tileSize = g->tileSize;
                unsigned int tilesPerDim = g->tilesPerDim;
                unsigned int src1 = srcIdx[e];
                unsigned int src2 = dst;
                unsigned int src1Tile = src1/tileSize;
                unsigned int src2Tile = src2/tileSize;
                unsigned int numTriangles_e = 0;
                for(unsigned int xTile = blockIdx.y; xTile < tilesPerDim && numTriangles_e < k - 2; xTile += gridDim.y) {
                    unsigned int tileSrc1 = (src1Tile*tilesPerDim + xTile)*tileSize + src1%tileSize;
                    unsigned int tileSrc2 = (src2Tile*tilesPerDim + xTile)*tileSize + src2%tileSize;
                    unsigned int e1 = tileSrcPtr[tileSrc1];
                    unsigned int e2 = tileSrcPtr[tileSrc2];
                    unsigned int end1 = tileSrcPtr[tileSrc1 + 1];
                    unsigned int end2 = tileSrcPtr[tileSrc2 + 1];
                    while(e1 < end1 && e2 < end2 && numTriangles_e < k - 2) {
                        unsigned int dst1 = dstIdx[e1];
                        if(CHECK_IF_DELETED && dst1 == DELETED) {
                            ++e1;
                        } else {
                            unsigned int dst2 = dstIdx[e2];
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
                                    ++numTriangles_e;
                                }
                            }
                        }
                    }
                }
                if(gridDim.y == 1) {
                    info.numTriangles[e] = numTriangles_e;
                } else {
                    atomicAdd(&info.numTriangles[e], numTriangles_e);
                }
            }
        }
    }
}

template < unsigned int CHECK_IF_DELETED, unsigned int RECOUNT_ALL_EDGES >
__global__ void mark_deleted_edges_kernel(TiledCOOCSRGraph* g, unsigned int k, Info info) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < g->numEdges) {
        if(RECOUNT_ALL_EDGES || info.edgeAffected[e] == DIRECTLY_AFFECTED) {
            unsigned int* dstIdx = g->dstIdx;
            unsigned int dst = dstIdx[e];
            if(!CHECK_IF_DELETED || dst != DELETED) {
                if(info.numTriangles[e] < k - 2) {
                    dstIdx[e] = DELETED;
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
__global__ void mark_directly_affected_edges_kernel(TiledCOOCSRGraph* g, Info info) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < g->numEdges) {
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

void ktruss(TiledCOOCSRGraph* g_d, Info info, Config config) {

    unsigned int k = config.k;
    unsigned int numEdges = config.numEdges;
    unsigned int iter = 0;
    unsigned int graphHasDeletedEdges = 0;
    unsigned int changed;
    initInfoOnDevice(info, config);
    do {

        if(config.verbosity >= 2) printf("        Iteration %u\n", iter);
        clearIterInfoOnDevice(info, config);

        // Count triangles
        Timer iterTimer = initTimer(config.verbosity >= 2);
        startTimer(&iterTimer);
        unsigned int numThreadsPerBlock = config.blockSize;
        unsigned int numBlocks = (numEdges + numThreadsPerBlock - 1)/numThreadsPerBlock;
        if(graphHasDeletedEdges) {
            if(config.recount == ALL) {
                count_triangles_kernel<1,1> <<< dim3(numBlocks, config.numParallelTiles), numThreadsPerBlock >>> (g_d, k, info);
            } else { // config.recount == AFFECTED
                count_triangles_kernel<1,0> <<< dim3(numBlocks, config.numParallelTiles), numThreadsPerBlock >>> (g_d, k, info);
            }
        } else {
            if(config.recount == ALL) {
                count_triangles_kernel<0,1> <<< dim3(numBlocks, config.numParallelTiles), numThreadsPerBlock >>> (g_d, k, info);
            } else { // config.recount == AFFECTED
                count_triangles_kernel<0,0> <<< dim3(numBlocks, config.numParallelTiles), numThreadsPerBlock >>> (g_d, k, info);
            }
        }
        syncStopAndPrintElapsed(&iterTimer, "            Triangle counting time");

        // Mark deleted edges
        startTimer(&iterTimer);
        if(graphHasDeletedEdges) {
            if(config.recount == ALL) {
                mark_deleted_edges_kernel<1,1> <<< numBlocks, numThreadsPerBlock >>> (g_d, k, info);
            } else { // config.recount == AFFECTED
                mark_deleted_edges_kernel<1,0> <<< numBlocks, numThreadsPerBlock >>> (g_d, k, info);
            }
        } else {
            if(config.recount == ALL) {
                mark_deleted_edges_kernel<0,1> <<< numBlocks, numThreadsPerBlock >>> (g_d, k, info);
            } else { // config.recount == AFFECTED
                mark_deleted_edges_kernel<0,0> <<< numBlocks, numThreadsPerBlock >>> (g_d, k, info);
            }
        }
        syncStopAndPrintElapsed(&iterTimer, "            Mark deleted edges");

        // Check if the graph changed
        cudaMemcpy(&changed, info.changed, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if(changed) {

            // Remove deleted edges
            if(iter < config.numEdgeRemoveIter) {
                startTimer(&iterTimer);
                removeTiledCOOCSRDeletedEdgesOnDevice(g_d);
                graphHasDeletedEdges = 0;
                cudaMemcpy(&numEdges, &g_d->numEdges, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                syncStopAndPrintElapsed(&iterTimer, "            Remove deleted edges");
                if(config.verbosity >= 2) printf("                # edges remaining = %u\n", numEdges);
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
                    mark_directly_affected_edges_kernel<1> <<< (numEdges + 1024 - 1)/1024, 1024 >>> (g_d, info);
                } else {
                    mark_directly_affected_edges_kernel<0> <<< (numEdges + 1024 - 1)/1024, 1024 >>> (g_d, info);
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
        removeTiledCOOCSRDeletedEdgesOnDevice(g_d);
        syncStopAndPrintElapsed(&timer, "        Remove deleted edges");
    }

}

void ktruss(COOGraph* graph_d, COOGraph* truss_d, Info info, Config config) {

    Timer timer = initTimer(config.verbosity >= 1);

    // Convert COO to tiled COOCSR
    startTimer(&timer);
    TiledCOOCSRGraph* g_d = createEmptyTiledCOOCSROnDevice(config.numNodes, config.numTiles, config.numEdges);
    coo2tiledcoocsrOnDevice(graph_d, g_d);
    syncStopAndPrintElapsed(&timer, "    Convert undirected COO to tiled COOCSR (not optimized)");

    // Runs
    for(unsigned int i = 0; i < config.numWarmupRuns + config.numTimedRuns; ++i) {

        // K-truss
        printAndStart(&timer, "    Performing K-truss\n");
        Timer ktrussTimer = initTimer(config.verbosity == 0 && i >= config.numWarmupRuns);
        startTimer(&ktrussTimer);
        ktruss(g_d, info, config);
        if(config.verbosity == 0 && i >= config.numWarmupRuns) printConfigAsCSV(config);
        syncStopAndPrintElapsed(&ktrussTimer);
        syncStopAndPrintElapsed(&timer, "        Total K-truss time", GREEN);

        // Restore graph
        if(i < config.numWarmupRuns + config.numTimedRuns - 1) {
            coo2tiledcoocsrOnDevice(graph_d, g_d);
            cudaDeviceSynchronize();
        }

    }

    // Convert tiled COOCSR to COO
    startTimer(&timer);
    tiledcoocsr2cooOnDevice(g_d, truss_d);
    syncStopAndPrintElapsed(&timer, "    Convert tiled COOCSR to COO");

    // Deallocate tiled COOCSR
    startTimer(&timer);
    freeTiledCOOCSRGraphOnDevice(g_d);
    syncStopAndPrintElapsed(&timer, "    Deallocate tiled COOCSR");

}

} } } // end namespace



// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "graph.h"
#include "kernel.h"
#include "timer.h"

namespace Device { namespace Directed { namespace COOCSRInput {

template < unsigned int CHECK_IF_DELETED , unsigned int RECOUNT_ALL_EDGES>
__global__ void count_triangles_kernel(COOCSRGraph* g, Info info) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < g->numEdges) {
        if(RECOUNT_ALL_EDGES || info.edgeAffected[e] == DIRECTLY_AFFECTED || info.edgeAffected[e] == INDIRECTLY_AFFECTED) {
            unsigned int* dstIdx = g->dstIdx;
            unsigned int* srcPtr = g->srcPtr;
            unsigned int dst = dstIdx[e];
            if(!CHECK_IF_DELETED || dst != DELETED) {
                unsigned int src1 = g->srcIdx[e];
                unsigned int src2 = dst;
                unsigned int e1 = srcPtr[src1];
                unsigned int e2 = srcPtr[src2];
                unsigned int end1 = srcPtr[src1 + 1];
                unsigned int end2 = srcPtr[src2 + 1];
                unsigned int numTriangles_e = 0;
                while(e1 < end1 && e2 < end2) {
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
                                ++numTriangles_e;
                                atomicAdd(&info.numTriangles[e1], 1);
                                atomicAdd(&info.numTriangles[e2], 1);
                                ++e1;
                                ++e2;
                            }
                        }
                    }
                }
                atomicAdd(&info.numTriangles[e], numTriangles_e);
            }
        }
    }
}

template < unsigned int CHECK_IF_DELETED, unsigned int RECOUNT_ALL_EDGES >
__global__ void mark_deleted_edges_kernel(COOCSRGraph* g, unsigned int k, Info info) {
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
__global__ void mark_directly_affected_edges_kernel(COOCSRGraph* g, Info info) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < g->numEdges) {
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
__global__ void mark_indirectly_affected_edges_kernel(COOCSRGraph* g, Info info) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < g->numEdges) {
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

void ktruss(COOCSRGraph* g_d, Info info, Config config) {

    unsigned int k = config.k;
    unsigned int numEdges = config.numEdges/2;
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
                count_triangles_kernel<1,1> <<< numBlocks, numThreadsPerBlock >>> (g_d, info);
            } else { // config.recount == AFFECTED
                count_triangles_kernel<1,0> <<< numBlocks, numThreadsPerBlock >>> (g_d, info);
            }
        } else {
            if(config.recount == ALL) {
                count_triangles_kernel<0,1> <<< numBlocks, numThreadsPerBlock >>> (g_d, info);
            } else { // config.recount == AFFECTED
                count_triangles_kernel<0,0> <<< numBlocks, numThreadsPerBlock >>> (g_d, info);
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
                removeCOOCSRDeletedEdgesOnDevice(g_d);
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
                    mark_indirectly_affected_edges_kernel<1> <<< (numEdges + 1024 - 1)/1024, 1024 >>> (g_d, info);
                } else {
                    mark_directly_affected_edges_kernel<0> <<< (numEdges + 1024 - 1)/1024, 1024 >>> (g_d, info);
                    mark_indirectly_affected_edges_kernel<0> <<< (numEdges + 1024 - 1)/1024, 1024 >>> (g_d, info);
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
        removeCOOCSRDeletedEdgesOnDevice(g_d);
        syncStopAndPrintElapsed(&timer, "        Remove deleted edges");
    }

}

void ktruss(COOGraph* gdir_d, COOGraph* truss_d, Info info, Config config) {

    Timer timer = initTimer(config.verbosity >= 1);

    // Convert COO to COOCSR
    startTimer(&timer);
    COOCSRGraph* g_d = createEmptyCOOCSROnDevice(config.numNodes, config.numEdges/2);
    coo2coocsrOnDevice(gdir_d, g_d);
    syncStopAndPrintElapsed(&timer, "    Convert directed COO to COOCSR (not optimized)");

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
            coo2coocsrOnDevice(gdir_d, g_d);
            cudaDeviceSynchronize();
        }

    }

    // Convert COOCSR to COO
    startTimer(&timer);
    coocsr2cooOnDevice(g_d, truss_d);
    syncStopAndPrintElapsed(&timer, "    Convert COO/CSR to COO");

    // Deallocate COOCSR
    startTimer(&timer);
    freeCOOCSRGraphOnDevice(g_d);
    syncStopAndPrintElapsed(&timer, "    Deallocate COOCSR");

}

} } } // end namespace


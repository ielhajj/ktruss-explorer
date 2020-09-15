
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "info.h"

Info createInfo(Config config) {
    Info info;
    unsigned int numEdges = (isDirected(config.directedness))?(config.numEdges/2):(config.numEdges);
    info.changed = (unsigned int*) malloc(sizeof(unsigned int));
    info.numTriangles = (unsigned int*) malloc(numEdges*sizeof(unsigned int));
    if(config.recount == AFFECTED) {
        info.nodeAffected = (unsigned int*) malloc(config.numNodes*sizeof(unsigned int));
        info.edgeAffected = (unsigned int*) malloc(numEdges*sizeof(unsigned int));
    } else {
        info.nodeAffected = NULL;
        info.edgeAffected = NULL;
    }
    return info;
}

Info createInfoOnDevice(Config config) {
    Info info;
    unsigned int numEdges = (isDirected(config.directedness))?(config.numEdges/2):(config.numEdges);
    cudaMalloc((void**) &info.changed, sizeof(unsigned int));
    cudaMalloc((void**) &info.numTriangles, numEdges*sizeof(unsigned int));
    if(config.recount == AFFECTED) {
        cudaMalloc((void**) &info.nodeAffected, config.numNodes*sizeof(unsigned int));
        cudaMalloc((void**) &info.edgeAffected, numEdges*sizeof(unsigned int));
    } else {
        info.nodeAffected = NULL;
        info.edgeAffected = NULL;
    }
    return info;
}

void initInfo(Info info, Config config) {
    unsigned int numEdges = (isDirected(config.directedness))?(config.numEdges/2):(config.numEdges);
    if(config.recount == AFFECTED) {
        for(unsigned int e = 0; e < numEdges; ++e) {
            // NOTE: Initially assume all edges are affected
            info.edgeAffected[e] = DIRECTLY_AFFECTED;
        }
    }
}

__global__ void init_edge_affected(unsigned int* edgeAffected, unsigned int numEdges) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < numEdges) {
        // NOTE: Initially assume all edges are affected
        edgeAffected[e] = DIRECTLY_AFFECTED;
    }
}

void initInfoOnDevice(Info info, Config config) {
    unsigned int numEdges = (isDirected(config.directedness))?(config.numEdges/2):(config.numEdges);
    if(config.recount == AFFECTED) {
        init_edge_affected <<< (numEdges + 1024 - 1)/1024, 1024 >>>(info.edgeAffected, numEdges);
    }
}

void clearIterInfo(Info info, Config config) {
    unsigned int numEdges = (isDirected(config.directedness))?(config.numEdges/2):(config.numEdges);
    *info.changed = 0;
    memset(info.numTriangles, 0, numEdges*sizeof(unsigned int));
    if(config.recount == AFFECTED) {
        memset(info.nodeAffected, NOT_AFFECTED, config.numNodes*sizeof(unsigned int));
        // NOTE: Affetced edges are not cleared because they are need it by next iteration
    }
}

void clearIterInfoOnDevice(Info info, Config config) {
    unsigned int numEdges = (isDirected(config.directedness))?(config.numEdges/2):(config.numEdges);
    cudaMemset(info.changed, 0, sizeof(unsigned int));
    cudaMemset(info.numTriangles, 0, numEdges*sizeof(unsigned int));
    if(config.recount == AFFECTED) {
        cudaMemset(info.nodeAffected, NOT_AFFECTED, config.numNodes*sizeof(unsigned int));
        // NOTE: Affetced edges are not cleared because they are need it by next iteration
    }
}

void freeInfo(Info info) {
    free(info.changed);
    free(info.numTriangles);
    if(info.nodeAffected != NULL) {
        free(info.nodeAffected);
    }
    if(info.edgeAffected != NULL) {
        free(info.edgeAffected);
    }
}

void freeInfoOnDevice(Info info) {
    cudaFree(info.changed);
    cudaFree(info.numTriangles);
    if(info.nodeAffected != NULL) {
        cudaFree(info.nodeAffected);
    }
    if(info.edgeAffected != NULL) {
        cudaFree(info.edgeAffected);
    }
}


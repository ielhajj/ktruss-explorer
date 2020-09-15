
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#include "graph.h"

#include <assert.h>
#include <stdio.h>

struct COOGraph* createEmptyCOO(unsigned int numNodes, unsigned int capacity) {
    struct COOGraph* cooGraph = (COOGraph*) malloc(sizeof(COOGraph));
    cooGraph->numNodes = numNodes;
    cooGraph->numEdges = 0;
    cooGraph->capacity = capacity;
    cooGraph->srcIdx = (unsigned int*) malloc(cooGraph->capacity*sizeof(unsigned int));
    cooGraph->dstIdx = (unsigned int*) malloc(cooGraph->capacity*sizeof(unsigned int));
    return cooGraph;
}

struct COOGraph* createEmptyCOOOnDevice(unsigned int numNodes, unsigned int capacity) {

    struct COOGraph g_shd;
    g_shd.numNodes = numNodes;
    g_shd.numEdges = 0;
    g_shd.capacity = capacity;
    cudaMalloc((void**) &g_shd.srcIdx, g_shd.capacity*sizeof(unsigned int));
    cudaMalloc((void**) &g_shd.dstIdx, g_shd.capacity*sizeof(unsigned int));

    struct COOGraph* g_d;
    cudaMalloc((void**) &g_d, sizeof(COOGraph));
    cudaMemcpy(g_d, &g_shd, sizeof(COOGraph), cudaMemcpyHostToDevice);

    return g_d;

}

struct COOGraph* createCOOFromFile(const char* fileName) {

    // Allocate
    struct COOGraph* cooGraph = (COOGraph*) malloc(sizeof(COOGraph));
    cooGraph->capacity = 1 << 20;
    cooGraph->srcIdx = (unsigned int*) malloc(cooGraph->capacity*sizeof(unsigned int));
    cooGraph->dstIdx = (unsigned int*) malloc(cooGraph->capacity*sizeof(unsigned int));

    // Read edges
    FILE* fp = fopen(fileName, "r");
    assert(fp != NULL);
    unsigned int numNodes = 0;
    unsigned int numEdges = 0;
    unsigned int src, dst, x;
    while(fscanf(fp, "%u", &dst) == 1) {
        assert(fscanf(fp, "%u", &src));
        assert(fscanf(fp, "%u", &x));
        assert(src != dst && "Edges from a vertex to itself are not allowed!");
        if(numEdges == cooGraph->capacity) {
            cooGraph->capacity = 2*cooGraph->capacity;
            cooGraph->srcIdx = (unsigned int*) realloc(cooGraph->srcIdx, cooGraph->capacity*sizeof(unsigned int));
            cooGraph->dstIdx = (unsigned int*) realloc(cooGraph->dstIdx, cooGraph->capacity*sizeof(unsigned int));
        }
        cooGraph->srcIdx[numEdges] = src - 1;
        cooGraph->dstIdx[numEdges] = dst - 1;
        ++numEdges;
        if(src > numNodes) {
            numNodes = src;
        }
        if(dst > numNodes) {
            numNodes = dst;
        }
    }

    // Update counts
    cooGraph->numNodes = numNodes;
    cooGraph->numEdges = numEdges;

    fclose(fp);

    return cooGraph;

}

void freeCOOGraph(struct COOGraph* cooGraph) {
    free(cooGraph->srcIdx);
    free(cooGraph->dstIdx);
    free(cooGraph);
}

void freeCOOGraphOnDevice(struct COOGraph* g_d) {
    struct COOGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(g_shd.srcIdx);
    cudaFree(g_shd.dstIdx);
    cudaFree(g_d);
}

void writeCOOGraphToFile(COOGraph* cooGraph, const char* fileName) {
    FILE* fp = fopen(fileName, "w");
    for(unsigned int e = 0; e < cooGraph->numEdges; ++e) {
        fprintf(fp, "%u\t", cooGraph->dstIdx[e] + 1);
        fprintf(fp, "%u\t", cooGraph->srcIdx[e] + 1);
        fprintf(fp, "%u\n", 1);
    }
    fclose(fp);
}

void copyCOOToDevice(struct COOGraph* g, struct COOGraph* g_d) {
    struct COOGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(g_shd.numNodes == g->numNodes);
    assert(g_shd.capacity >= g->numEdges);
    cudaMemcpy(&g_d->numEdges, &g->numEdges, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_shd.srcIdx, g->srcIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_shd.dstIdx, g->dstIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
}

void copyCOOFromDevice(struct COOGraph* g_d, struct COOGraph* g) {
    struct COOGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(g->numNodes == g_shd.numNodes);
    assert(g->capacity >= g_shd.numEdges);
    g->numEdges = g_shd.numEdges;
    cudaMemcpy(g->srcIdx, g_shd.srcIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(g->dstIdx, g_shd.dstIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void quicksort(unsigned int *key, unsigned int* data, unsigned int start, unsigned int end) {
    if((end - start + 1) > 1) {
        unsigned int left = start, right = end;
        unsigned int pivot = key[right];
        while(left <= right) {
            while(key[left] < pivot) {
                left = left + 1;
            }
            while(key[right] > pivot) {
                right = right - 1;
            }
            if(left <= right) {
                unsigned int tmpKey = key[left]; key[left] = key[right]; key[right] = tmpKey;
                unsigned int tmpData = data[left]; data[left] = data[right]; data[right] = tmpData;
                left = left + 1;
                right = right - 1;
            }
        }
        quicksort(key, data, start, right);
        quicksort(key, data, left, end);
    }
}

void sortByDegree(COOGraph* graph, unsigned int* new2old) {

    // Initialize permutation vector
    for(unsigned int i = 0; i < graph->numNodes; ++i) {
        new2old[i] = i;
    }

    // Find degree of each node
    unsigned int* degree = (unsigned int*) calloc(graph->numNodes, sizeof(unsigned int));
    for(unsigned int e = 0; e < graph->numEdges; ++e) {
        degree[graph->srcIdx[e]]++;
    }

    // Sort nodes by degree
    quicksort(degree, new2old, 0, graph->numNodes - 1);

    // Find inverse permutation
    unsigned int* old2new = (unsigned int*) malloc(graph->numNodes*sizeof(unsigned int));
    for(unsigned int newIdx = 0; newIdx < graph->numNodes; ++newIdx) {
        unsigned int oldIdx = new2old[newIdx];
        old2new[oldIdx] = newIdx;
    }

    // Update edges
    for(unsigned int e = 0; e < graph->numEdges; ++e) {
        graph->srcIdx[e] = old2new[graph->srcIdx[e]];
        graph->dstIdx[e] = old2new[graph->dstIdx[e]];
    }

    // Free intermediate data
    free(degree);
    free(old2new);

}

void unsort(COOGraph* graph, unsigned int* new2old) {
    for(unsigned int e = 0; e < graph->numEdges; ++e) {
        graph->srcIdx[e] = new2old[graph->srcIdx[e]];
        graph->dstIdx[e] = new2old[graph->dstIdx[e]];
    }
}

void undirected2directedCOO(struct COOGraph* gundirected, struct COOGraph* gdirected) {
    assert(gdirected->numNodes == gundirected->numNodes);
    gdirected->numEdges = 0;
    for(unsigned int e = 0; e < gundirected->numEdges; ++e) {
        unsigned int src = gundirected->srcIdx[e];
        unsigned int dst = gundirected->dstIdx[e];
        if(src < dst) {
            unsigned int eout = gdirected->numEdges++;
            assert(eout < gdirected->capacity);
            gdirected->srcIdx[eout] = src;
            gdirected->dstIdx[eout] = dst;
        }
    }
}

__global__ void undirected2directedCOO_kernel(struct COOGraph* gundirected, struct COOGraph* gdirected) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < gundirected->numEdges) {
        unsigned int src = gundirected->srcIdx[e];
        unsigned int dst = gundirected->dstIdx[e];
        if(src < dst) {
            unsigned int eout = atomicAdd(&gdirected->numEdges, 1);
            gdirected->srcIdx[eout] = src;
            gdirected->dstIdx[eout] = dst;
        }
    }
}

void undirected2directedCOOOnDevice(struct COOGraph* gundirected_d, struct COOGraph* gdirected_d) {

    // Copy shadows from device
    COOGraph gundirected_shd;
    COOGraph gdirected_shd;
    cudaMemcpy(&gundirected_shd, gundirected_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gdirected_shd, gdirected_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Convert
    assert(gdirected_shd.numNodes == gundirected_shd.numNodes);
    assert(gdirected_shd.capacity >= gundirected_shd.numEdges/2);
    cudaMemset(&gdirected_d->numEdges, 0, sizeof(unsigned int));
    undirected2directedCOO_kernel <<< (gundirected_shd.numEdges + 1024 - 1)/1024, 1024 >>> (gundirected_d, gdirected_d);

}

void directed2undirectedCOO(struct COOGraph* g) {
    unsigned int numDirectedEdges = g->numEdges;
    for(unsigned int e = 0; e < numDirectedEdges; ++e) {
        unsigned int src = g->srcIdx[e];
        unsigned int dst = g->dstIdx[e];
        if(src != dst) {
            unsigned int eout = g->numEdges++;
            assert(eout < g->capacity);
            g->srcIdx[eout] = dst;
            g->dstIdx[eout] = src;
        }
    }
}


__global__ void directed2undirectedCOO_kernel(struct COOGraph* g, unsigned int numDirectedEdges) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < numDirectedEdges) {
        unsigned int src = g->srcIdx[e];
        unsigned int dst = g->dstIdx[e];
        if(src != dst) {
            unsigned int eout = atomicAdd(&g->numEdges, 1);
            g->srcIdx[eout] = dst;
            g->dstIdx[eout] = src;
        }
    }
}

void directed2undirectedCOOOnDevice(struct COOGraph* g_d) {

    // Copy shadow from device
    COOGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Convert
    assert(g_shd.capacity >= 2*g_shd.numEdges);
    directed2undirectedCOO_kernel <<< (g_shd.numEdges + 1024 - 1)/1024, 1024 >>> (g_d, g_shd.numEdges);

}

void quicksort(unsigned int *key, unsigned int start, unsigned int end) {
    if((end - start + 1) > 1) {
        unsigned int left = start, right = end;
        unsigned int pivot = key[right];
        while(left <= right) {
            while(key[left] < pivot) {
                left = left + 1;
            }
            while(key[right] > pivot) {
                right = right - 1;
            }
            if(left <= right) {
                unsigned int tmpKey = key[left]; key[left] = key[right]; key[right] = tmpKey;
                left = left + 1;
                right = right - 1;
            }
        }
        quicksort(key, start, right);
        quicksort(key, left, end);
    }
}

struct COOCSRGraph* createEmptyCOOCSR(unsigned int numNodes, unsigned int capacity) {
    struct COOCSRGraph* graph = (COOCSRGraph*) malloc(sizeof(COOCSRGraph));
    graph->numNodes = numNodes;
    graph->numEdges = 0;
    graph->capacity = capacity;
    graph->srcPtr = (unsigned int*) malloc((graph->numNodes + 1)*sizeof(unsigned int));
    graph->srcIdx = (unsigned int*) malloc(graph->capacity*sizeof(unsigned int));
    graph->dstIdx = (unsigned int*) malloc(graph->capacity*sizeof(unsigned int));
    return graph;
}

struct COOCSRGraph* createEmptyCOOCSROnDevice(unsigned int numNodes, unsigned int capacity) {

    struct COOCSRGraph g_shd;
    g_shd.numNodes = numNodes;
    g_shd.numEdges = 0;
    g_shd.capacity = capacity;
    cudaMalloc((void**) &g_shd.srcPtr, (g_shd.numNodes + 1)*sizeof(unsigned int));
    cudaMalloc((void**) &g_shd.srcIdx, g_shd.capacity*sizeof(unsigned int));
    cudaMalloc((void**) &g_shd.dstIdx, g_shd.capacity*sizeof(unsigned int));

    struct COOCSRGraph* g_d;
    cudaMalloc((void**) &g_d, sizeof(COOCSRGraph));
    cudaMemcpy(g_d, &g_shd, sizeof(COOCSRGraph), cudaMemcpyHostToDevice);

    return g_d;

}

void freeCOOCSRGraph(struct COOCSRGraph* graph) {
    free(graph->srcPtr);
    free(graph->srcIdx);
    free(graph->dstIdx);
    free(graph);
}

void freeCOOCSRGraphOnDevice(struct COOCSRGraph* g_d) {
    struct COOCSRGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(COOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(g_shd.srcPtr);
    cudaFree(g_shd.srcIdx);
    cudaFree(g_shd.dstIdx);
    cudaFree(g_d);
}

void copyCOOCSRToDevice(struct COOCSRGraph* g, struct COOCSRGraph* g_d) {
    struct COOCSRGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(COOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(g_shd.numNodes == g->numNodes);
    assert(g_shd.capacity >= g->numEdges);
    cudaMemcpy(&g_d->numEdges, &g->numEdges, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_shd.srcPtr, g->srcPtr, (g->numNodes + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_shd.srcIdx, g->srcIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_shd.dstIdx, g->dstIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
}

void copyCOOCSRFromDevice(struct COOCSRGraph* g_d, struct COOCSRGraph* g) {
    struct COOCSRGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(COOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(g->numNodes == g_shd.numNodes);
    assert(g->capacity >= g_shd.numEdges);
    g->numEdges = g_shd.numEdges;
    cudaMemcpy(g->srcPtr, g_shd.srcPtr, (g->numNodes + 1)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(g->srcIdx, g_shd.srcIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(g->dstIdx, g_shd.dstIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

void coo2coocsr(struct COOGraph* cooGraph, struct COOCSRGraph* graph) {

    // Initialize
    unsigned int numNodes = cooGraph->numNodes;
    assert(graph->numNodes == numNodes);
    unsigned int numEdges = cooGraph->numEdges;
    assert(graph->capacity >= numEdges);
    graph->numEdges = numEdges;

    // Histogram
    // NOTE: (src + 1) used instead of (src) because it will get shifted by the binning operation
    memset(graph->srcPtr, 0, (numNodes + 1)*sizeof(unsigned int));
    for(unsigned int e = 0; e < numEdges; ++e) {
        unsigned int src = cooGraph->srcIdx[e];
        graph->srcPtr[src + 1]++;
    }

    // Prefix sum
    unsigned int sum = 0;
    for(unsigned int src = 0; src < numNodes; ++src) {
        unsigned int val = graph->srcPtr[src + 1];
        graph->srcPtr[src + 1] = sum;
        sum += val;
    }

    // Binning
    for(unsigned int e = 0; e < numEdges; ++e) {
        unsigned int src = cooGraph->srcIdx[e];
        unsigned int j = graph->srcPtr[src + 1]++;
        graph->srcIdx[j] = src;
        graph->dstIdx[j] = cooGraph->dstIdx[e];
    }

    // Sort outgoing edges of each source node
    for(unsigned int src = 0; src < numNodes; ++src) {
        unsigned int start = graph->srcPtr[src];
        unsigned int end = graph->srcPtr[src + 1] - 1;
        quicksort(graph->dstIdx, start, end); // NOTE: No need to sort srcIdx because they are all the same
    }

}

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

__global__ void histogram_kernel(unsigned int* data, unsigned int* bins, unsigned int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        unsigned int val = data[i];
        atomicAdd(&bins[val], 1);
    }
}

__global__ void binning_kernel(COOGraph* cooGraph, COOCSRGraph* graph) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < cooGraph->numEdges) {
        unsigned int src = cooGraph->srcIdx[e];
        unsigned int j = atomicAdd(&graph->srcPtr[src + 1], 1);
        graph->srcIdx[j] = src;
        graph->dstIdx[j] = cooGraph->dstIdx[e];
    }
}

void coo2coocsrOnDevice(struct COOGraph* cooGraph_d, struct COOCSRGraph* graph_d) {

    // Copy shadows from device
    COOGraph cooGraph_shd;
    COOCSRGraph graph_shd;
    cudaMemcpy(&cooGraph_shd, cooGraph_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaMemcpy(&graph_shd, graph_d, sizeof(COOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Initialize
    unsigned int numNodes = cooGraph_shd.numNodes;
    assert(graph_shd.numNodes == numNodes);
    unsigned int numEdges = cooGraph_shd.numEdges;
    assert(graph_shd.capacity >= numEdges);
    cudaMemcpy(&graph_d->numEdges, &cooGraph_shd.numEdges, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Histogram
    // NOTE: (srcPtr + 1) used instead of (srcPtr) because it will get shifted by the binning operation
    cudaMemset(graph_shd.srcPtr, 0, (numNodes + 1)*sizeof(unsigned int));
    histogram_kernel <<< (numEdges + 1024 - 1)/1024, 1024 >>> (cooGraph_shd.srcIdx, graph_shd.srcPtr + 1, numEdges);

    // Prefix sum
    thrust::exclusive_scan(thrust::device, graph_shd.srcPtr + 1, graph_shd.srcPtr + numNodes + 1, graph_shd.srcPtr + 1);

    // Binning
    binning_kernel <<< (numEdges + 1024 - 1)/1024, 1024 >>> (cooGraph_d, graph_d);

    // Sort outgoing edges of each source node (on CPU)
    // TODO: Implement sorting on GPU
    unsigned int* srcPtr = (unsigned int*) malloc((numNodes + 1)*sizeof(unsigned int));
    unsigned int* dstIdx = (unsigned int*) malloc(numEdges*sizeof(unsigned int));
    cudaMemcpy(srcPtr, graph_shd.srcPtr, (numNodes + 1)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dstIdx, graph_shd.dstIdx, numEdges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(unsigned int src = 0; src < numNodes; ++src) {
        unsigned int start = srcPtr[src];
        unsigned int end = srcPtr[src + 1] - 1;
        quicksort(dstIdx, start, end); // NOTE: No need to sort srcIdx because they are all the same
    }
    cudaMemcpy(graph_shd.dstIdx, dstIdx, numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    free(srcPtr);
    free(dstIdx);

}

void removeCOOCSRDeletedEdges(struct COOCSRGraph* g) {

    // Compact edges
    unsigned int oldNumEdges = g->numEdges;
    g->numEdges = 0;
    for(unsigned int e = 0; e < oldNumEdges; ++e) {
        if(g->dstIdx[e] != DELETED) {
            g->srcIdx[g->numEdges] = g->srcIdx[e];
            g->dstIdx[g->numEdges] = g->dstIdx[e];
            g->numEdges++;
        }
    }

    // Histogram
    memset(g->srcPtr, 0, (g->numNodes + 1)*sizeof(unsigned int));
    for(unsigned int e = 0; e < g->numEdges; ++e) {
        unsigned int src = g->srcIdx[e];
        g->srcPtr[src]++;
    }

    // Prefix sum
    unsigned int sum = 0;
    for(unsigned int src = 0; src < g->numNodes; ++src) {
        unsigned int val = g->srcPtr[src];
        g->srcPtr[src] = sum;
        sum += val;
    }
    g->srcPtr[g->numNodes] = sum;

}

__global__ void mark_deleted_srcs_kernel(COOCSRGraph* g) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < g->numEdges) {
        if(g->dstIdx[e] == DELETED) {
            g->srcIdx[e] = DELETED;
        }
    }
}

void removeCOOCSRDeletedEdgesOnDevice(struct COOCSRGraph* g_d) {

    // Copy shadow
    COOCSRGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(COOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Mark deleted sources
    mark_deleted_srcs_kernel <<< (g_shd.numEdges + 1024 - 1)/1024, 1024 >>> (g_d);

    // Compact edges
    unsigned int* endSrcIdx = thrust::remove(thrust::device, g_shd.srcIdx, g_shd.srcIdx + g_shd.numEdges, DELETED);
    unsigned int* endDstIdx = thrust::remove(thrust::device, g_shd.dstIdx, g_shd.dstIdx + g_shd.numEdges, DELETED);
    assert(endSrcIdx - g_shd.srcIdx == endDstIdx - g_shd.dstIdx);
    g_shd.numEdges = endSrcIdx - g_shd.srcIdx;
    cudaMemcpy(&g_d->numEdges, &g_shd.numEdges, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Histogram
    unsigned int numNodes = g_shd.numNodes;
    unsigned int numEdges = g_shd.numEdges;
    cudaMemset(g_shd.srcPtr, 0, (numNodes + 1)*sizeof(unsigned int));
    histogram_kernel <<< (numEdges + 1024 - 1)/1024, 1024 >>> (g_shd.srcIdx, g_shd.srcPtr, numEdges);

    // Prefix sum
    thrust::exclusive_scan(thrust::device, g_shd.srcPtr, g_shd.srcPtr + numNodes + 1, g_shd.srcPtr);

}

void coocsr2coo(struct COOCSRGraph* in, struct COOGraph* out) {
    assert(out->numNodes == in->numNodes);
    assert(out->capacity >= in->numEdges);
    out->numEdges = in->numEdges;
    memcpy(out->srcIdx, in->srcIdx, in->numEdges*sizeof(unsigned int));
    memcpy(out->dstIdx, in->dstIdx, in->numEdges*sizeof(unsigned int));
}

void coocsr2cooOnDevice(struct COOCSRGraph* in_d, struct COOGraph* out_d) {

    // Copy shadows from device
    COOCSRGraph in_shd;
    COOGraph out_shd;
    cudaMemcpy(&in_shd, in_d, sizeof(COOCSRGraph), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_shd, out_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Convert
    assert(out_shd.numNodes == in_shd.numNodes);
    assert(out_shd.capacity >= in_shd.numEdges);
    cudaMemcpy(&out_d->numEdges, &in_shd.numEdges, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_shd.srcIdx, in_shd.srcIdx, in_shd.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_shd.dstIdx, in_shd.dstIdx, in_shd.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);

}

struct TiledCOOCSRGraph* createEmptyTiledCOOCSR(unsigned int numNodes, unsigned int tilesPerDim, unsigned int capacity) {
    struct TiledCOOCSRGraph* graph = (TiledCOOCSRGraph*) malloc(sizeof(TiledCOOCSRGraph));
    graph->numNodes = numNodes;
    graph->numEdges = 0;
    graph->capacity = capacity;
    graph->tilesPerDim = tilesPerDim;
    graph->tileSize = (numNodes + tilesPerDim - 1)/tilesPerDim;
    unsigned int numTileSrcPtrs = tilesPerDim*tilesPerDim*graph->tileSize + 1;
    graph->tileSrcPtr = (unsigned int*) malloc(numTileSrcPtrs*sizeof(unsigned int));
    graph->srcIdx = (unsigned int*) malloc(graph->capacity*sizeof(unsigned int));
    graph->dstIdx = (unsigned int*) malloc(graph->capacity*sizeof(unsigned int));
    return graph;
}

struct TiledCOOCSRGraph* createEmptyTiledCOOCSROnDevice(unsigned int numNodes, unsigned int tilesPerDim, unsigned int capacity) {

    struct TiledCOOCSRGraph g_shd;
    g_shd.numNodes = numNodes;
    g_shd.numEdges = 0;
    g_shd.capacity = capacity;
    g_shd.tilesPerDim = tilesPerDim;
    g_shd.tileSize = (numNodes + tilesPerDim - 1)/tilesPerDim;
    unsigned int numTileSrcPtrs = tilesPerDim*tilesPerDim*g_shd.tileSize + 1;
    cudaMalloc((void**) &g_shd.tileSrcPtr, numTileSrcPtrs*sizeof(unsigned int));
    cudaMalloc((void**) &g_shd.srcIdx, g_shd.capacity*sizeof(unsigned int));
    cudaMalloc((void**) &g_shd.dstIdx, g_shd.capacity*sizeof(unsigned int));

    struct TiledCOOCSRGraph* g_d;
    cudaMalloc((void**) &g_d, sizeof(TiledCOOCSRGraph));
    cudaMemcpy(g_d, &g_shd, sizeof(TiledCOOCSRGraph), cudaMemcpyHostToDevice);

    return g_d;

}

void freeTiledCOOCSRGraph(struct TiledCOOCSRGraph* graph) {
    free(graph->tileSrcPtr);
    free(graph->srcIdx);
    free(graph->dstIdx);
    free(graph);
}

void freeTiledCOOCSRGraphOnDevice(struct TiledCOOCSRGraph* g_d) {
    struct TiledCOOCSRGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(TiledCOOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(g_shd.tileSrcPtr);
    cudaFree(g_shd.srcIdx);
    cudaFree(g_shd.dstIdx);
    cudaFree(g_d);
}

void copyTiledCOOCSRToDevice(struct TiledCOOCSRGraph* g, struct TiledCOOCSRGraph* g_d) {
    struct TiledCOOCSRGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(TiledCOOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(g_shd.numNodes == g->numNodes);
    assert(g_shd.capacity >= g->numEdges);
    assert(g_shd.tilesPerDim == g->tilesPerDim);
    assert(g_shd.tileSize == g->tileSize);
    unsigned int numTileSrcPtrs = g->tilesPerDim*g->tilesPerDim*g->tileSize + 1;
    cudaMemcpy(&g_d->numEdges, &g->numEdges, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_shd.tileSrcPtr, g->tileSrcPtr, numTileSrcPtrs*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_shd.srcIdx, g->srcIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_shd.dstIdx, g->dstIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
}

void copyTiledCOOCSRFromDevice(struct TiledCOOCSRGraph* g_d, struct TiledCOOCSRGraph* g) {
    struct TiledCOOCSRGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(TiledCOOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(g->numNodes == g_shd.numNodes);
    assert(g->capacity >= g_shd.numEdges);
    assert(g->tilesPerDim == g_shd.tilesPerDim);
    assert(g->tileSize == g_shd.tileSize);
    g->numEdges = g_shd.numEdges;
    unsigned int numTileSrcPtrs = g->tilesPerDim*g->tilesPerDim*g->tileSize + 1;
    cudaMemcpy(g->tileSrcPtr, g_shd.tileSrcPtr, numTileSrcPtrs*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(g->srcIdx, g_shd.srcIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(g->dstIdx, g_shd.dstIdx, g->numEdges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

void coo2tiledcoocsr(struct COOGraph* cooGraph, struct TiledCOOCSRGraph* graph) {

    // Initialize
    unsigned int numNodes = cooGraph->numNodes;
    assert(graph->numNodes == numNodes);
    unsigned int numEdges = cooGraph->numEdges;
    assert(graph->capacity >= numEdges);
    graph->numEdges = numEdges;

    // Histogram
    // NOTE: (tileSrc + 1) used instead of (tileSrc) because it will get shifted by the binning operation
    unsigned int tileSize = graph->tileSize;
    unsigned int tilesPerDim = graph->tilesPerDim;
    unsigned int numTileSrcPtrs = tilesPerDim*tilesPerDim*tileSize;
    memset(graph->tileSrcPtr, 0, (numTileSrcPtrs + 1)*sizeof(unsigned int));
    for(unsigned int e = 0; e < numEdges; ++e) {
        unsigned int src = cooGraph->srcIdx[e];
        unsigned int dst = cooGraph->dstIdx[e];
        unsigned int srcTile = src/tileSize;
        unsigned int dstTile = dst/tileSize;
        unsigned int tileSrc = (srcTile*tilesPerDim + dstTile)*tileSize + src%tileSize;
        graph->tileSrcPtr[tileSrc + 1]++;
    }

    // Prefix sum
    unsigned int sum = 0;
    for(unsigned int tileSrc = 0; tileSrc < numTileSrcPtrs; ++tileSrc) {
        unsigned int val = graph->tileSrcPtr[tileSrc + 1];
        graph->tileSrcPtr[tileSrc + 1] = sum;
        sum += val;
    }

    // Binning
    for(unsigned int e = 0; e < numEdges; ++e) {
        unsigned int src = cooGraph->srcIdx[e];
        unsigned int dst = cooGraph->dstIdx[e];
        unsigned int srcTile = src/tileSize;
        unsigned int dstTile = dst/tileSize;
        unsigned int tileSrc = (srcTile*tilesPerDim + dstTile)*tileSize + src%tileSize;
        unsigned int j = graph->tileSrcPtr[tileSrc + 1]++;
        graph->srcIdx[j] = src;
        graph->dstIdx[j] = cooGraph->dstIdx[e];
    }

    // Sort outgoing edges of each source node
    for(unsigned int tileSrc = 0; tileSrc < numTileSrcPtrs; ++tileSrc) {
        unsigned int start = graph->tileSrcPtr[tileSrc];
        unsigned int end = graph->tileSrcPtr[tileSrc + 1] - 1;
        quicksort(graph->dstIdx, start, end); // NOTE: No need to sort srcIdx because they are all the same
    }

}

__global__ void histogram_tiled_kernel(COOGraph* cooGraph, TiledCOOCSRGraph* graph) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < cooGraph->numEdges) {
        unsigned int src = cooGraph->srcIdx[e];
        unsigned int dst = cooGraph->dstIdx[e];
        unsigned int tileSize = graph->tileSize;
        unsigned int tilesPerDim = graph->tilesPerDim;
        unsigned int tileSrc = (src/tileSize*tilesPerDim + dst/tileSize)*tileSize + src%tileSize;
        atomicAdd(&graph->tileSrcPtr[tileSrc + 1], 1);
    }
}

__global__ void binning_kernel(COOGraph* cooGraph, TiledCOOCSRGraph* graph) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < cooGraph->numEdges) {
        unsigned int src = cooGraph->srcIdx[e];
        unsigned int dst = cooGraph->dstIdx[e];
        unsigned int tileSize = graph->tileSize;
        unsigned int tilesPerDim = graph->tilesPerDim;
        unsigned int tileSrc = (src/tileSize*tilesPerDim + dst/tileSize)*tileSize + src%tileSize;
        unsigned int j = atomicAdd(&graph->tileSrcPtr[tileSrc + 1], 1);
        graph->srcIdx[j] = src;
        graph->dstIdx[j] = cooGraph->dstIdx[e];
    }
}

void coo2tiledcoocsrOnDevice(struct COOGraph* cooGraph_d, struct TiledCOOCSRGraph* graph_d) {

    // Copy shadows from device
    COOGraph cooGraph_shd;
    TiledCOOCSRGraph graph_shd;
    cudaMemcpy(&cooGraph_shd, cooGraph_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaMemcpy(&graph_shd, graph_d, sizeof(TiledCOOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Initialize
    unsigned int numNodes = cooGraph_shd.numNodes;
    assert(graph_shd.numNodes == numNodes);
    unsigned int numEdges = cooGraph_shd.numEdges;
    assert(graph_shd.capacity >= numEdges);
    cudaMemcpy(&graph_d->numEdges, &cooGraph_shd.numEdges, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Histogram
    // NOTE: (tileSrc + 1) used instead of (tileSrc) because it will get shifted by the binning operation
    unsigned int tilesPerDim = graph_shd.tilesPerDim;
    unsigned int numTileSrcPtrs = tilesPerDim*tilesPerDim*graph_shd.tileSize;
    cudaMemset(graph_shd.tileSrcPtr, 0, (numTileSrcPtrs + 1)*sizeof(unsigned int));
    histogram_tiled_kernel <<< (numEdges + 1024 - 1)/1024, 1024 >>> (cooGraph_d, graph_d);

    // Prefix sum
    thrust::exclusive_scan(thrust::device, graph_shd.tileSrcPtr + 1, graph_shd.tileSrcPtr + numTileSrcPtrs + 1, graph_shd.tileSrcPtr + 1);

    // Binning
    binning_kernel <<< (numEdges + 1024 - 1)/1024, 1024 >>> (cooGraph_d, graph_d);

    // Sort outgoing edges of each source node (on CPU)
    // TODO: Implement sorting on GPU
    unsigned int* tileSrcPtr = (unsigned int*) malloc((numTileSrcPtrs + 1)*sizeof(unsigned int));
    unsigned int* dstIdx = (unsigned int*) malloc(numEdges*sizeof(unsigned int));
    cudaMemcpy(tileSrcPtr, graph_shd.tileSrcPtr, (numTileSrcPtrs + 1)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dstIdx, graph_shd.dstIdx, numEdges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(unsigned int tileSrc = 0; tileSrc < numTileSrcPtrs; ++tileSrc) {
        unsigned int start = tileSrcPtr[tileSrc];
        unsigned int end = tileSrcPtr[tileSrc + 1] - 1;
        quicksort(dstIdx, start, end); // NOTE: No need to sort srcIdx because they are all the same
    }
    cudaMemcpy(graph_shd.dstIdx, dstIdx, numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    free(tileSrcPtr);
    free(dstIdx);

}

void removeTiledCOOCSRDeletedEdges(struct TiledCOOCSRGraph* g) {

    // Compact edges
    unsigned int oldNumEdges = g->numEdges;
    g->numEdges = 0;
    for(unsigned int e = 0; e < oldNumEdges; ++e) {
        if(g->dstIdx[e] != DELETED) {
            g->srcIdx[g->numEdges] = g->srcIdx[e];
            g->dstIdx[g->numEdges] = g->dstIdx[e];
            g->numEdges++;
        }
    }

    // Histogram
    unsigned int tileSize = g->tileSize;
    unsigned int tilesPerDim = g->tilesPerDim;
    unsigned int numTileSrcPtrs = tilesPerDim*tilesPerDim*tileSize;
    memset(g->tileSrcPtr, 0, (numTileSrcPtrs + 1)*sizeof(unsigned int));
    for(unsigned int e = 0; e < g->numEdges; ++e) {
        unsigned int src = g->srcIdx[e];
        unsigned int dst = g->dstIdx[e];
        unsigned int srcTile = src/tileSize;
        unsigned int dstTile = dst/tileSize;
        unsigned int tileSrc = (srcTile*tilesPerDim + dstTile)*tileSize + src%tileSize;
        g->tileSrcPtr[tileSrc]++;
    }

    // Prefix sum
    unsigned int sum = 0;
    for(unsigned int tileSrc = 0; tileSrc < numTileSrcPtrs; ++tileSrc) {
        unsigned int val = g->tileSrcPtr[tileSrc];
        g->tileSrcPtr[tileSrc] = sum;
        sum += val;
    }
    g->tileSrcPtr[numTileSrcPtrs] = sum;

}

__global__ void histogram_tiled_remove_kernel(TiledCOOCSRGraph* g) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < g->numEdges) {
        unsigned int src = g->srcIdx[e];
        unsigned int dst = g->dstIdx[e];
        unsigned int tileSize = g->tileSize;
        unsigned int tilesPerDim = g->tilesPerDim;
        unsigned int tileSrc = (src/tileSize*tilesPerDim + dst/tileSize)*tileSize + src%tileSize;
        atomicAdd(&g->tileSrcPtr[tileSrc], 1);
    }
}

__global__ void mark_deleted_srcs_tiled_kernel(TiledCOOCSRGraph* g) {
    unsigned int e = blockIdx.x*blockDim.x + threadIdx.x;
    if(e < g->numEdges) {
        if(g->dstIdx[e] == DELETED) {
            g->srcIdx[e] = DELETED;
        }
    }
}

void removeTiledCOOCSRDeletedEdgesOnDevice(struct TiledCOOCSRGraph* g_d) {

    // Copy shadow
    TiledCOOCSRGraph g_shd;
    cudaMemcpy(&g_shd, g_d, sizeof(TiledCOOCSRGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Mark deleted sources
    mark_deleted_srcs_tiled_kernel <<< (g_shd.numEdges + 1024 - 1)/1024, 1024 >>> (g_d);

    // Compact edges
    unsigned int* endSrcIdx = thrust::remove(thrust::device, g_shd.srcIdx, g_shd.srcIdx + g_shd.numEdges, DELETED);
    unsigned int* endDstIdx = thrust::remove(thrust::device, g_shd.dstIdx, g_shd.dstIdx + g_shd.numEdges, DELETED);
    assert(endSrcIdx - g_shd.srcIdx == endDstIdx - g_shd.dstIdx);
    g_shd.numEdges = endSrcIdx - g_shd.srcIdx;
    cudaMemcpy(&g_d->numEdges, &g_shd.numEdges, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Histogram
    unsigned int numEdges = g_shd.numEdges;
    unsigned int tileSize = g_shd.tileSize;
    unsigned int tilesPerDim = g_shd.tilesPerDim;
    unsigned int numTileSrcPtrs = tilesPerDim*tilesPerDim*tileSize;
    cudaMemset(g_shd.tileSrcPtr, 0, (numTileSrcPtrs + 1)*sizeof(unsigned int));
    histogram_tiled_remove_kernel <<< (numEdges + 1024 - 1)/1024, 1024 >>> (g_d);

    // Prefix sum
    thrust::exclusive_scan(thrust::device, g_shd.tileSrcPtr, g_shd.tileSrcPtr + numTileSrcPtrs + 1, g_shd.tileSrcPtr);

}

void tiledcoocsr2coo(struct TiledCOOCSRGraph* in, struct COOGraph* out) {
    assert(out->numNodes == in->numNodes);
    assert(out->capacity >= in->numEdges);
    out->numEdges = in->numEdges;
    memcpy(out->srcIdx, in->srcIdx, in->numEdges*sizeof(unsigned int));
    memcpy(out->dstIdx, in->dstIdx, in->numEdges*sizeof(unsigned int));
}

void tiledcoocsr2cooOnDevice(struct TiledCOOCSRGraph* in_d, struct COOGraph* out_d) {

    // Copy shadows from device
    TiledCOOCSRGraph in_shd;
    COOGraph out_shd;
    cudaMemcpy(&in_shd, in_d, sizeof(TiledCOOCSRGraph), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_shd, out_d, sizeof(COOGraph), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Convert
    assert(out_shd.numNodes == in_shd.numNodes);
    assert(out_shd.capacity >= in_shd.numEdges);
    cudaMemcpy(&out_d->numEdges, &in_shd.numEdges, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_shd.srcIdx, in_shd.srcIdx, in_shd.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_shd.dstIdx, in_shd.dstIdx, in_shd.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);

}


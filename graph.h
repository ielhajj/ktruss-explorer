
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#ifndef _GRAPH_H_
#define _GRAPH_H_

#define DELETED UINT_MAX

struct COOGraph {
    unsigned int numNodes;
    unsigned int numEdges;
    unsigned int capacity;
    unsigned int* srcIdx;
    unsigned int* dstIdx;
};

struct COOGraph* createEmptyCOO(unsigned int numNodes, unsigned int capacity);
struct COOGraph* createEmptyCOOOnDevice(unsigned int numNodes, unsigned int capacity);
struct COOGraph* createCOOFromFile(const char* fileName);

void freeCOOGraph(struct COOGraph* cooGraph);
void freeCOOGraphOnDevice(struct COOGraph* cooGraph);

void writeCOOGraphToFile(COOGraph* cooGraph, const char* fileName);

void copyCOOToDevice(struct COOGraph* g, struct COOGraph* g_d);
void copyCOOFromDevice(struct COOGraph* g_d, struct COOGraph* g);

void sortByDegree(COOGraph* graph, unsigned int* new2old);

void unsort(COOGraph* graph, unsigned int* new2old);

void undirected2directedCOO(struct COOGraph* gundirected, struct COOGraph* gdirected);
void undirected2directedCOOOnDevice(struct COOGraph* gundirected, struct COOGraph* gdirected);

void directed2undirectedCOO(struct COOGraph* g);
void directed2undirectedCOOOnDevice(struct COOGraph* g);

struct COOCSRGraph {
    unsigned int numNodes;
    unsigned int numEdges;
    unsigned int capacity;
    unsigned int* srcPtr;
    unsigned int* srcIdx;
    unsigned int* dstIdx;
};

struct COOCSRGraph* createEmptyCOOCSR(unsigned int numNodes, unsigned int capacity);
struct COOCSRGraph* createEmptyCOOCSROnDevice(unsigned int numNodes, unsigned int capacity);

void freeCOOCSRGraph(struct COOCSRGraph* graph);
void freeCOOCSRGraphOnDevice(struct COOCSRGraph* g_d);

void copyCOOCSRToDevice(struct COOCSRGraph* g, struct COOCSRGraph* g_d);
void copyCOOCSRFromDevice(struct COOCSRGraph* g_d, struct COOCSRGraph* g);

void coo2coocsr(struct COOGraph* cooGraph, struct COOCSRGraph* graph);
void coo2coocsrOnDevice(struct COOGraph* cooGraph_d, struct COOCSRGraph* graph_d);

void removeCOOCSRDeletedEdges(struct COOCSRGraph* g);
void removeCOOCSRDeletedEdgesOnDevice(struct COOCSRGraph* g_d);

void coocsr2coo(struct COOCSRGraph* in, struct COOGraph* out);
void coocsr2cooOnDevice(struct COOCSRGraph* in_d, struct COOGraph* out_d);

struct TiledCOOCSRGraph {
    unsigned int numNodes;
    unsigned int numEdges;
    unsigned int tilesPerDim;
    unsigned int tileSize;
    unsigned int capacity;
    unsigned int* tileSrcPtr;
    unsigned int* srcIdx;
    unsigned int* dstIdx;
};

struct TiledCOOCSRGraph* createEmptyTiledCOOCSR(unsigned int numNodes, unsigned int tilesPerDim, unsigned int capacity);
struct TiledCOOCSRGraph* createEmptyTiledCOOCSROnDevice(unsigned int numNodes, unsigned int tilesPerDim, unsigned int capacity);

void freeTiledCOOCSRGraph(struct TiledCOOCSRGraph* graph);
void freeTiledCOOCSRGraphOnDevice(struct TiledCOOCSRGraph* g_d);

void copyTiledCOOCSRToDevice(struct TiledCOOCSRGraph* g, struct TiledCOOCSRGraph* g_d);
void copyTiledCOOCSRFromDevice(struct TiledCOOCSRGraph* g_d, struct TiledCOOCSRGraph* g);

void coo2tiledcoocsr(struct COOGraph* cooGraph, struct TiledCOOCSRGraph* graph);
void coo2tiledcoocsrOnDevice(struct COOGraph* cooGraph_d, struct TiledCOOCSRGraph* graph_d);

void removeTiledCOOCSRDeletedEdges(struct TiledCOOCSRGraph* g);
void removeTiledCOOCSRDeletedEdgesOnDevice(struct TiledCOOCSRGraph* g_d);

void tiledcoocsr2coo(struct TiledCOOCSRGraph* in, struct COOGraph* out);
void tiledcoocsr2cooOnDevice(struct TiledCOOCSRGraph* in_d, struct COOGraph* out_d);

#endif


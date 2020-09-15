
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <unistd.h>
#include <stdio.h>

static void usage() {
    fprintf(stderr,
            "\nUsage:  ./ktruss [options]"
            "\n"
            "\nOptions:"
            "\n    -g <graphFileName>       Name of file with input graph (default = data/loc-gowalla_edges_adj.tsv)"
            "\n    -k <k>                   k (default = 3)"
            "\n    -c <compareFileName>     Name of file to compare to when verifying result (default = data/loc-gowalla_edges_adj.tsv.3.reference.tsv)"
            "\n    -o <outFileName>         Name of ouput file to write result to (by default, the result will not be written to a file)"
            "\n    -w <numWarmupRuns>       Number of warmup runs (default = 0)"
            "\n    -x <numTimedRuns>        Number of timed runs (default = 1)"
            "\n    -p <processor>           Processor to run on. Options:"
            "\n                                 cpu - CPU"
            "\n                                 gpu - GPU (default)"
            "\n    -d <directedness>        Directedness of graph. Options:"
            "\n                                 undirected - use undirected graph as is"
            "\n                                 directed-by-index - only keep edges from vertices with lower index to vertices with higher index"
            "\n                                 directed-by-degree - only keep edges from lower degree vertices to higher degree vertices (default)"
            "\n    -f <inputFormat>         Format of the graph. Options:"
            "\n                                 coocsr - convert to hybrid COO/CSR graph before processing (default)"
            "\n                                 tiled-coocsr - convert to hybrid COO/CSR graph with tiling before processing"
            "\n    -e <numEdgeRemoveIter>   Number of initial iterations during which edges are to be removed (default = 0)"
            "\n    -r <recount>             Which edges to recount triangles for. Options:"
            "\n                                 all - recount triangles for all undeleted edges every iteration (default)"
            "\n                                 affected - recount triangles for only edges affected by deletions in the previous iteration"
            "\n    -b <blockSize>           Number of threads per block in k-truss kernel (default = 512)"
            "\n                                 NOTE: -b only has effect with option: -p gpu"
            "\n    -t <numTiles>            Number of tiles for tiled COO/CSR (default = 8)"
            "\n                                 NOTE: -t only has effect with option: -f tiled-coocsr"
            "\n    -y <numParallelTiles>    Number of tiles for tiled COO/CSR to process in parallel on the GPU (default = 8)"
            "\n                                 NOTE: -y only has effect with options: -p gpu -f tiled-coocsr"
            "\n    -v <verbosity>           Options:"
            "\n                                 0 - show k-truss time only in CSV format"
            "\n                                 1 - show allocation, copy, and conversion times (default)"
            "\n                                 2 - show per-iteration times"
            "\n    -h                       Help"
            "\n"
            "\n");
}

enum Processor {
    CPU,
    GPU
};

static Processor parseProcessor(const char* s) {
    if(strcmp(s, "cpu") == 0) {
        return CPU;
    } else if(strcmp(s, "gpu") == 0) {
        return GPU;
    } else {
        fprintf(stderr, "Unrecognized -p option: %s\n", s);
        exit(0);
    }
}

static const char* asString(Processor processor) {
    switch(processor) {
        case CPU: return "cpu";
        case GPU: return "gpu";
        default:
            fprintf(stderr, "Unrecognized processor\n");
            exit(0);
    }
}

enum Directedness {
    UNDIRECTED,
    DIRECTED_BY_INDEX,
    DIRECTED_BY_DEGREE
};

static Directedness parseDirectedness(const char* s) {
    if(strcmp(s, "undirected") == 0) {
        return UNDIRECTED;
    } else if(strcmp(s, "directed-by-index") == 0) {
        return DIRECTED_BY_INDEX;
    } else if(strcmp(s, "directed-by-degree") == 0) {
        return DIRECTED_BY_DEGREE;
    } else {
        fprintf(stderr, "Unrecognized -d option: %s\n", s);
        exit(0);
    }
}

static const char* asString(Directedness directedness) {
    switch(directedness) {
        case UNDIRECTED:            return "undirected";
        case DIRECTED_BY_INDEX:     return "directed-by-index";
        case DIRECTED_BY_DEGREE:    return "directed-by-degree";
        default:
            fprintf(stderr, "Unrecognized directedness\n");
            exit(0);
    }
}

static unsigned int isDirected(Directedness directedness) {
    return (directedness == DIRECTED_BY_INDEX) || (directedness == DIRECTED_BY_DEGREE);
}

enum InputFormat {
    COOCSR,
    TILED_COOCSR
};

static InputFormat parseInputFormat(const char* s) {
    if(strcmp(s, "coocsr") == 0) {
        return COOCSR;
    } else if(strcmp(s, "tiled-coocsr") == 0) {
        return TILED_COOCSR;
    } else {
        fprintf(stderr, "Unrecognized -f option: %s\n", s);
        exit(0);
    }
}

static const char* asString(InputFormat inputFormat) {
    switch(inputFormat) {
        case COOCSR:        return "coocsr";
        case TILED_COOCSR:  return "tiled-coocsr";
        default:
            fprintf(stderr, "Unrecognized input format\n");
            exit(0);
    }
}

enum Recount {
    ALL,
    AFFECTED
};

static Recount parseRecount(const char* s) {
    if(strcmp(s, "all") == 0) {
        return ALL;
    } else if(strcmp(s, "affected") == 0) {
        return AFFECTED;
    } else {
        fprintf(stderr, "Unrecognized -r option: %s\n", s);
        exit(0);
    }
}

static const char* asString(Recount recount) {
    switch(recount) {
        case ALL     : return "all";
        case AFFECTED: return "affected";
        default:
            fprintf(stderr, "Unrecognized edge removal\n");
            exit(0);
    }
}

struct Config {
    const char* graphFileName;
    unsigned int k;
    const char* compareFileName;
    const char* outFileName;
    unsigned int numWarmupRuns;
    unsigned int numTimedRuns;
    Processor processor;
    Directedness directedness;
    InputFormat inputFormat;
    unsigned int numEdgeRemoveIter;
    Recount recount;
    unsigned int blockSize;
    unsigned int numTiles;
    unsigned int numParallelTiles;
    unsigned int verbosity;
    unsigned int numNodes; // Initialized when graph is read
    unsigned int numEdges; // Initialized when graph is read
};

static Config parseArgs(int argc, char** argv) {
    Config config;
    config.graphFileName = "data/loc-gowalla_edges_adj.tsv";
    config.k = 3;
    config.compareFileName = "data/loc-gowalla_edges_adj.tsv.3.reference.tsv";
    config.outFileName = NULL;
    config.numWarmupRuns = 0;
    config.numTimedRuns = 1;
    config.processor = GPU;
    config.directedness = DIRECTED_BY_DEGREE;
    config.inputFormat = COOCSR;
    config.numEdgeRemoveIter = 0;
    config.recount = ALL;
    config.blockSize = 512;
    config.numTiles = 8;
    config.numParallelTiles = 8;
    config.verbosity = 1;
    int opt;
    while((opt = getopt(argc, argv, "g:k:o:w:x:c:p:d:f:e:r:b:t:y:v:h")) >= 0) {
        switch(opt) {
            case 'g': config.graphFileName      = optarg;                           break;
            case 'k': config.k                  = atoi(optarg);                     break;
            case 'c': config.compareFileName    = optarg;                           break;
            case 'o': config.outFileName        = optarg;                           break;
            case 'w': config.numWarmupRuns      = atoi(optarg);                     break;
            case 'x': config.numTimedRuns       = atoi(optarg);                     break;
            case 'p': config.processor          = parseProcessor(optarg);           break;
            case 'd': config.directedness       = parseDirectedness(optarg);        break;
            case 'f': config.inputFormat        = parseInputFormat(optarg);         break;
            case 'e': config.numEdgeRemoveIter  = atoi(optarg);                     break;
            case 'r': config.recount            = parseRecount(optarg);             break;
            case 'b': config.blockSize          = atoi(optarg);                     break;
            case 't': config.numTiles           = atoi(optarg);                     break;
            case 'y': config.numParallelTiles   = atoi(optarg);                     break;
            case 'v': config.verbosity          = atoi(optarg);                     break;
            case 'h': usage(); exit(0);
            default : fprintf(stderr, "\nUnrecognized option!\n");
                      usage(); exit(0);
        }
    }
    return config;
}

static void printConfig(Config config) {
    printf("    Graph: %s\n", config.graphFileName);
    printf("        # vertices = %u\n", config.numNodes);
    printf("        # edges = %u\n", config.numEdges);
    printf("    k: %u\n", config.k);
    printf("    Processor: %s\n", asString(config.processor));
    if(config.processor == GPU) {
        printf("        Block size = %u\n", config.blockSize);
    }
    printf("    Directedness: %s\n", asString(config.directedness));
    printf("    Input format: %s\n", asString(config.inputFormat));
    if(config.inputFormat == TILED_COOCSR) {
        printf("        Number of tiles = %u\n", config.numTiles);
        if(config.processor == GPU) {
            printf("        Number of tiles processed in parallel = %u\n", config.numParallelTiles);
        }
    }
    printf("    Number of initial iterations to remove edges: %u\n", config.numEdgeRemoveIter);
    printf("    Recount strategy: %s\n", asString(config.recount));
}

static void printConfigAsCSV(Config config) {
    printf("%s, ", config.graphFileName);
    printf("%u, ", config.numNodes);
    printf("%u, ", config.numEdges);
    printf("%u, ", config.k);
    printf("%s, ", asString(config.processor));
    printf("%s, ", asString(config.directedness));
    printf("%s, ", asString(config.inputFormat));
    printf("%u, ", config.numEdgeRemoveIter);
    printf("%s, ", asString(config.recount));
    if(config.processor == GPU) {
        printf("%u, ", config.blockSize);
    } else {
        printf("N/A, ");
    }
    if(config.inputFormat == TILED_COOCSR) {
        printf("%u, ", config.numTiles);
        if(config.processor == GPU) {
            printf("%u, ", config.numParallelTiles);
        } else {
            printf("N/A, ");
        }
    } else {
        printf("N/A, N/A, ");
    }
}

#endif


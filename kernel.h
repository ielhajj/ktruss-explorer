
// Copyright (c) 2020, American University of Beirut
// See LICENSE.txt for copyright license

#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "config.h"
#include "graph.h"
#include "info.h"

void ktruss(COOGraph* graph, COOGraph* truss, Config config);

namespace Host {
    void ktruss(COOGraph* graph, COOGraph* truss, Config config);
    namespace Undirected {
        namespace COOCSRInput {
            void ktruss(COOGraph* graph, COOGraph* truss, Info info, Config config);
        }
        namespace TiledCOOCSRInput {
            void ktruss(COOGraph* graph, COOGraph* truss, Info info, Config config);
        }
    }
    namespace Directed {
        namespace COOCSRInput {
            void ktruss(COOGraph* gdir, COOGraph* truss, Info info, Config config);
        }
        namespace TiledCOOCSRInput {
            void ktruss(COOGraph* gdir, COOGraph* truss, Info info, Config config);
        }
    }
}

namespace Device {
    void ktruss(COOGraph* graph, COOGraph* truss, Config config);
    namespace Undirected {
        namespace COOCSRInput {
            void ktruss(COOGraph* graph_d, COOGraph* truss_d, Info info, Config config);
        }
        namespace TiledCOOCSRInput {
            void ktruss(COOGraph* graph_d, COOGraph* truss_d, Info info, Config config);
        }
    }
    namespace Directed {
        namespace COOCSRInput {
            void ktruss(COOGraph* gdir_d, COOGraph* truss_d, Info info, Config config);
        }
        namespace TiledCOOCSRInput {
            void ktruss(COOGraph* gdir_d, COOGraph* truss_d, Info info, Config config);
        }
    }
}

#endif


/*
 * Copyright 2016 neurodata (http://neurodata.io/)
 * Written by Disa Mhembere (disa@jhu.edu)
 *
 * This file is part of k-par-means
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY CURRENT_KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

#include "dist_task_coordinator.hpp"
#include "dist_task_thread.hpp"
#include "clusters.hpp"
#include "mpi.h"

namespace kpmeans { namespace prune {

dist_task_coordinator::dist_task_coordinator(
        const std::string fn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const kpmbase::init_type_t it,
        const double tolerance, const kpmbase::dist_type_t dt) :
    kmeans_task_coordinator(fn, nrow, ncol, k, max_iters,
            nnodes, nthreads, centers, it, tolerance, dt) {

        // TODO
    }

void dist_task_coordinator::random_partition_init() {
    // TODO
}

void kmeanspp_init() {}
void forgy_init() {}
void run_kmeans() {}

/**
 * Main driver for kmeans
 */
void kmeans_task_coordinator::run_kmeans() {}

void aggregate_clusters() {}
void reduce_assignment_counts() {}
void merge_global_assignments() {}

} } // End namespace kpmeans, prune

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

#include "dist_coordinator.hpp"
#include "kmeans_thread.hpp"
#include "clusters.hpp"
#include "io.hpp"

namespace kpmeans {

dist_coordinator::dist_coordinator(
        const std::string fn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const unsigned mpi_rank, const unsigned nprocs,
        const double* centers, const kpmbase::init_type_t it,
        const double tolerance, const kpmbase::dist_type_t dt) :
    kmeans_coordinator(fn, get_proc_rows(nrow, nprocs, mpi_rank),
            ncol, k, max_iters, nnodes, nthreads, centers, it, tolerance, dt) {

        this->mpi_rank = mpi_rank;
        this->nprocs = nprocs;
        this->g_nrow = nrow;

        for (thread_iter it = threads.begin(); it < threads.end(); ++it)
            (*it)->set_start_rid((*it)->get_start_rid()
                    + (nrow / nprocs) * mpi_rank);
}

/**
  * This takes the global number of samples in the *entire* dataset, `g_nrow'
  *     and gives the coordinator it's partion.
  */
size_t const dist_coordinator::get_proc_rows(const size_t g_nrow,
        const unsigned nprocs, const unsigned mpi_rank) const {
    if (mpi_rank == (nprocs - 1)) // The last proc always has more
        return (g_nrow / nprocs) + (g_nrow % nprocs);
    else
        return (g_nrow / nprocs);
}

void dist_coordinator::random_partition_init() {
    kpmbase::rand123emulator<unsigned> gen(0, k-1,
            ((g_nrow / nprocs) * mpi_rank));
    for (size_t row = 0; row < nrow; row++) {
        unsigned asgnd_clust = gen.next();
        const double* dp = this->get_thd_data(row);

        cltrs->add_member(dp, asgnd_clust);
        cluster_assignments[row] = asgnd_clust;
    }

#if VERBOSE
    printf("After rand paritions cluster_asgns: ");
    print_arr<unsigned>(cluster_assignments, nrow);
#endif
}

const size_t dist_coordinator::global_rid(const size_t local_rid) const {
    return ((g_nrow / nprocs)*mpi_rank) + local_rid;
}

const size_t dist_coordinator::local_rid(const size_t global_rid) const {
    size_t rid = global_rid - (mpi_rank * (g_nrow / nprocs));
    if (rid > this->nrow)
        throw kpmbase::thread_exception("Row: " + std::to_string(rid) +
                " out of bounds for Proc: " + std::to_string( mpi_rank));
    return rid;
}

// For testing
void const dist_coordinator::print_thread_data() {
    std::cout << "\n\nProcess: " << this->mpi_rank;
    kmeans_coordinator::print_thread_data();
}

void dist_coordinator::kmeanspp_init() {
    throw kpmbase::not_implemented_exception();
}
void dist_coordinator::forgy_init() {
    throw kpmbase::not_implemented_exception();
}

void dist_coordinator::run_kmeans() {
    throw kpmbase::not_implemented_exception();
}

// Aggregate per process from threads &
//      save to `cltrs' as the delta for 1 EM-step
void dist_coordinator::pp_aggregate() {
    num_changed = 0; // Reset every iteration
    cltrs->clear(); // NOTE: So we don't clear prev_means

    for (thread_iter it = threads.begin(); it != threads.end(); ++it) {
        // Updated the changed cluster count
        num_changed += (*it)->get_num_changed();
        cltrs->peq((*it)->get_local_clusters());
    }
}
}

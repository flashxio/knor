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
#include "mpi.hpp"

namespace kpmmpi = kpmeans::mpi;

namespace kpmeans { namespace dist {

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
    kpmbase::print_arr<unsigned>(cluster_assignments, nrow);
#endif
}

/**
  * We need to shift the `start_rid` to be only local to this process so
  *     that `EM_step` in the threading class assigns to a global_rid that is
  *     local to only this process.
  */
void dist_coordinator::shift_thread_start_rid() {
    size_t c = 0;
    for (thread_iter it = threads.begin(); it != threads.end(); ++it) {
        size_t shift = (*it)->get_start_rid() - ((g_nrow / nprocs) * mpi_rank);
#if VERBOSE
        printf("P: %u, T: %lu, start_rid: %lu\n", mpi_rank, c++, shift);
#endif
        (*it)->set_start_rid(shift);
    }
}

const size_t dist_coordinator::global_rid(const size_t local_rid) const {
    return ((g_nrow / nprocs) * mpi_rank) + local_rid;
}

const size_t dist_coordinator::local_rid(const size_t global_rid) const {
    size_t rid = global_rid - (mpi_rank * (g_nrow / nprocs));
    if (rid > this->nrow)
        throw kpmbase::thread_exception("Row: " + std::to_string(rid) +
                " out of bounds for Proc: " + std::to_string( mpi_rank));
    return rid;
}

const bool dist_coordinator::is_local(const size_t global_rid) const {
    size_t rid = global_rid - (mpi_rank * (g_nrow / nprocs));
    if (rid >= this->nrow)
        return false;
    return true;
}

// For testing
void const dist_coordinator::print_thread_data() {
    std::cout << "\n\nProcess: " << this->mpi_rank;
    kmeans_coordinator::print_thread_data();
}

void dist_coordinator::kmeanspp_init() {
    struct timeval start, end;

    std::vector<double> buff(k*ncol);
    std::vector<double> dist_v;
    std::vector<double> g_dist_v(g_nrow);
    dist_v.assign(get_nrow(), std::numeric_limits<double>::max()); // local nrow
    set_thd_dist_v_ptr(&dist_v[0]);

    // Choose c1 uniformly at random
    unsigned selected_idx = random() % g_nrow; // 0...(g_nrow-1)

    // If proc owns the row -- get it ...
    if (is_local(selected_idx)) {
        cltrs->set_mean(get_thd_data(local_rid(selected_idx)), 0);
        dist_v[local_rid(selected_idx)] = 0.0;
        cluster_assignments[local_rid(selected_idx)] = 0;
    }

    kpmmpi::mpi::reduce_double(&(cltrs->get_means()[0]),
            &buff[0], cltrs->size());
    cltrs->set_mean(&buff[0]);

#if VERBOSE
    BOOST_LOG_TRIVIAL(info) << "Choosing "
        << selected_idx << " as center k = 0";
#endif
    unsigned clust_idx = 0; // The number of clusters assigned

    // Choose next center c_i with weighted prob
    while (true) {
        set_thread_clust_idx(clust_idx); // Set the current cluster index
        wake4run(KMSPP_INIT); // Run || distance comp to clust_idx
        wait4complete();
        double local_cuml_dist = reduction_on_cuml_sum(); // Per proc cuml dists

        double cuml_dist; // Recepticle
        kpmmpi::mpi::reduce_double(&local_cuml_dist, &cuml_dist);

        // All procs do this ...
        cuml_dist = (cuml_dist * ((double)random())) / (RAND_MAX - 1.0);
        if (++clust_idx >= k)  // No more centers needed
            break;

        // Gather the g_dist_v
        kpmmpi::mpi::allgather_double(&dist_v[0],
                &g_dist_v[0], g_nrow/nprocs);

        // Gather the remaining entries from the last proc which *may* have more
        if (g_nrow % nprocs) {
            const size_t tail_idx = (g_nrow/nprocs);
            const size_t numel = (g_nrow % nprocs);

            if (mpi_rank == nprocs - 1)
                std::copy(&dist_v[tail_idx], &dist_v[tail_idx+numel],
                        &g_dist_v[g_nrow-numel]);

            kpmmpi::mpi::bcast_double(&g_dist_v[g_nrow-numel], nprocs-1, numel);
        }

        for (size_t row = 0; row < g_nrow; row++) {
            cuml_dist -= g_dist_v[row];
            if (cuml_dist <= 0) {
#if VERBOSE
                if (mpi_rank == 0)
                    BOOST_LOG_TRIVIAL(info) << "Choosing "
                        << row << " as center k = " << clust_idx;
#endif

                if (is_local(row)) {
                    cltrs->set_mean(get_thd_data(local_rid(row)), clust_idx);
                    cluster_assignments[local_rid(row)] = clust_idx;
                } else {
                    cltrs->clear();
                }

                break;
            }
        }

        kpmmpi::mpi::reduce_double(&(cltrs->get_means()[0]),
                &buff[0], cltrs->size());
        cltrs->set_mean(&buff[0]);
        BOOST_VERIFY(cuml_dist <= 0);
    }

#if VERBOSE
    if (mpi_rank == 0) {
        BOOST_LOG_TRIVIAL(info) << "\nCluster centers after kmeans++";
        cltrs->print_means();
    }
#endif
    gettimeofday(&end, NULL);
    if (mpi_rank == 0)
        BOOST_LOG_TRIVIAL(info) << "Initialization time: " <<
            kpmbase::time_diff(start, end) << " sec\n";
}

void dist_coordinator::forgy_init() {
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(0, g_nrow-1);

    for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) { // 0...k
        size_t gid = distribution(generator);
        if (is_local(gid))
            cltrs->set_mean(get_thd_data(local_rid(gid)), clust_idx);
    }
}

kpmbase::kmeans_t dist_coordinator::run_kmeans() {
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

} } // End namespace kpmeans::dist

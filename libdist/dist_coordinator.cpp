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
#include "util.hpp"
#include "kmeans_types.hpp"

namespace kpmmpi = kpmeans::mpi;

namespace kpmeans { namespace dist {

dist_coordinator::dist_coordinator(
        int argc, char* argv[],
        const std::string fn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const kpmbase::init_type_t it,
        const double tolerance, const kpmbase::dist_type_t dt) :
    kmeans_coordinator(fn, this->init(argc, argv, nrow), ncol, k, max_iters, nnodes,
            nthreads, centers, it, tolerance, dt) {

        this->g_nrow = nrow;

        for (thread_iter it = threads.begin(); it < threads.end(); ++it)
            (*it)->set_start_rid((*it)->get_start_rid()
                    + (nrow / nprocs) * mpi_rank);
}

/**
  * A method called prior to calling the superclass constructor.
  * This takes the global number of samples in the *entire* dataset, `g_nrow'
  *     and gives the coordinator it's partition.
  * \param g_nrow: the global number of rows
  * \return the local number of rows for this process
  */
const size_t dist_coordinator::init(int argc, char* argv[],
        const size_t g_nrow) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Init error\n");

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // Set the num_procs
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

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
#ifndef BIND
    printf("After rand paritions cluster_asgns: \n");
#endif
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
#ifndef BIND
        printf("P: %u, T: %lu, start_rid: %lu\n", mpi_rank, c++, shift);
#endif
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

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, g_nrow-1);

    // Choose c1 uniformly at random
    unsigned selected_idx = distribution(generator);

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
#ifndef BIND
    printf("Choosing %u as center k = 0\n", selected_idx);
#endif
#endif
    unsigned clust_idx = 0; // The number of clusters assigned

    std::uniform_real_distribution<double> ur_distribution(0.0, 1.0);

    // Choose next center c_i with weighted prob
    while (true) {
        set_thread_clust_idx(clust_idx); // Set the current cluster index
        wake4run(KMSPP_INIT); // Run || distance comp to clust_idx
        wait4complete();
        double local_cuml_dist = reduction_on_cuml_sum(); // Per proc cuml dists

        double cuml_dist; // Recepticle
        kpmmpi::mpi::reduce_double(&local_cuml_dist, &cuml_dist);

        // All procs do this ...
        cuml_dist = (cuml_dist * ur_distribution(generator)) / (RAND_MAX - 1.0);
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
#ifndef BIND
                    printf("Choosing %lu as center k = %u\n", row, clust_idx);
#endif
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
        assert(cuml_dist <= 0);
    }

#if VERBOSE
    if (mpi_rank == 0) {
#ifndef BIND
        printf("\nCluster centers after kmeans++\n");
#endif
        cltrs->print_means();
    }
#endif
    gettimeofday(&end, NULL);
    if (mpi_rank == 0)
#ifndef BIND
        printf("Initialization time: %.6f sec\n",
                kpmbase::time_diff(start, end));
#endif
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

void dist_coordinator::run_kmeans(kpmbase::kmeans_t& ret,
        const std::string outdir) {
    if (mpi_rank == root) {
        if (outdir.empty())
            fprintf(stderr, "\n**[WARNING]**: No output dir specified with "
                    "'-o' flag means no output will be saved!\n");

#ifndef BIND
        printf("Running FULL kmeans\n");
#endif
    }

    // Give processes their data
    wake4run(kpmeans::thread_state_t::ALLOC_DATA);
    wait4complete();

    shift_thread_start_rid();

    struct timeval start, end;
    gettimeofday(&start , NULL);

    // Var init
    double perc_changed = std::numeric_limits<double>::max();
    bool converged = false;
    size_t iters = 0;
    size_t nchanged = 0;

    kpmbase::clusters::ptr cltrs_ptr = get_gcltrs();

    // Init
    run_init();

    double* clstr_buff = new double[k*ncol];
    size_t* nmemb_buff = new size_t[k];

    if (_init_t == kpmbase::init_type_t::RANDOM ||
            _init_t == kpmbase::init_type_t::FORGY) {
        // MPI Update clusters
        kpmmpi::mpi::reduce_double(&(cltrs_ptr->get_means()[0]),
                clstr_buff, cltrs_ptr->size());
        cltrs_ptr->set_mean(clstr_buff);

        if (_init_t == kpmbase::init_type_t::RANDOM) {
            kpmmpi::mpi::reduce_size_t(&(cltrs_ptr->get_num_members_v()[0]),
                    nmemb_buff, cltrs_ptr->get_num_members_v().size());
            cltrs_ptr->set_num_members_v(nmemb_buff); // Set new counts
            cltrs_ptr->finalize_all();
            // End Init

#ifdef VERBOSE
            assert((size_t)std::accumulate(cltrs_ptr->get_num_members_v().begin(),
                        cltrs_ptr->get_num_members_v().end(), 0) == g_nrow);
            if (mpi_rank == root) {
#ifndef BIND
                printf("New finalized centers for Proc: %d ==> \n", mpi_rank);
#endif
                cltrs_ptr->print_means();
            }
#endif
        }
    }

    // EM-step iterations
    while (iters < max_iters && max_iters > 0) {
        if (iters == 0)
            clear_cluster_assignments();

        // Init iteration
        if (mpi_rank == root)
#ifndef BIND
            printf("Running iteration %lu ...\n", iters);
#endif

        wake4run(kpmeans::thread_state_t::EM);
        wait4complete();
        // NOTE: Unfinalized diffs on this proc

        pp_aggregate();

        // NOTE: cltrs_ptr has this procs diff (agg of threads from this proc)
        // NOTE: clstr_buff has agg of all procs diff
        kpmmpi::mpi::reduce_double(&(cltrs_ptr->get_means()[0]),
                clstr_buff, cltrs_ptr->size());

        // nmemb_buff has agg of all procs diff on membership count
        kpmmpi::mpi::reduce_size_t(&(cltrs_ptr->get_num_members_v()[0]),
                nmemb_buff, cltrs_ptr->get_num_members_v().size());
        cltrs_ptr->set_mean(clstr_buff);
        cltrs_ptr->set_num_members_v(nmemb_buff);

        // NOTE: Now finalized
        size_t pp_num_changed = get_num_changed();
        kpmmpi::mpi::reduce_size_t(&pp_num_changed, &nchanged);

        if (mpi_rank == root) {
#ifndef BIND
            printf("Global nchanged: %lu ...\n", nchanged);
#endif
            cltrs_ptr->print_membership_count();
        }

        assert((size_t)std::accumulate(
                    cltrs_ptr->get_num_members_v().begin(),
                    cltrs_ptr->get_num_members_v().end(), 0) == g_nrow);

        perc_changed = (double)nchanged/g_nrow; // Global perc change
        cltrs_ptr->finalize_all();

        if (nchanged == 0 || perc_changed <= tolerance) {
            converged = true;
            if (mpi_rank == root)
#ifndef BIND
                printf("Algorithm converged in %lu iterations!\n", (++iters));
#endif
            break;
        }

        nchanged = 0;
        iters++;
    }

    if (!converged && mpi_rank == root)
#ifndef BIND
        printf("Algorithm failed to converge in %lu iterations\n", iters);
#endif

    gettimeofday(&end, NULL);
    if (mpi_rank == root)
#ifndef BIND
        printf("\nAlgorithmic time taken = %.5f sec\n",
                kpmbase::time_diff(start, end));
#endif

    if (!outdir.empty()) {
        // Collect cluster assignments
        const unsigned* local_assignments = get_cluster_assignments();

        if (mpi_rank != root) {
            int rc = MPI_Ssend(local_assignments, get_nrow(),
                    MPI::UNSIGNED, root, 0, MPI_COMM_WORLD);
            kpmbase::assert_msg(!rc,
                    "Failure to send local assignments to root");
        } else {
            std::vector<unsigned> assignments(g_nrow);
            std::copy(local_assignments, local_assignments+get_nrow(),
                    assignments.begin());

            for (int pid = 1; pid < int(nprocs); pid++) {
                // Account for an uneven # of rows per process
                unsigned count;
                if (pid == (nprocs - 1))
                    count = get_nrow() + (g_nrow % nprocs);
                else
                    count = get_nrow();

                int rc = MPI_Recv(&assignments[pid*(g_nrow/nprocs)],
                        count, MPI::UNSIGNED, pid,
                        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                kpmbase::assert_msg(!rc,
                        "Root Failure receive local assignments");
            }

            ret = kpmbase::kmeans_t(g_nrow, ncol, iters, k, &assignments[0],
                    &(cltrs_ptr->get_num_members_v()[0]),
                    cltrs_ptr->get_means());

            if (!outdir.empty()) {
#ifndef BIND
                printf("\nWriting output to '%s'\n", outdir.c_str());
#endif
                ret.write(outdir);
            }
        }
    }

    // MPI cleanup and graceful exit
    delete [] clstr_buff;
    delete [] nmemb_buff;
}

dist_coordinator::~dist_coordinator() {
    MPI_Finalize();
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

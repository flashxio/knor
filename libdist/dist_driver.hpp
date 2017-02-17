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

#ifndef __KPM_DIST_DRIVER_HPP__
#define __KPM_DIST_DRIVER_HPP__
#include <numa.h>

#include "exception.hpp"
#include "thread_state.hpp"
#include "dist_coordinator.hpp"
#include "util.hpp"
#include "mpi.hpp"
#include "io.hpp"
#include "clusters.hpp"

namespace kpmmpi = kpmeans::mpi;

namespace kpmeans { namespace dist {
class driver {

public:
static void run_kmeans(int argc, char* argv[],
        const std::string datafn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthread, kpmbase::kmeans_t& ret,
        const double* p_centers=NULL, const std::string init="kmeanspp",
        const double tolerance=-1, const std::string dist_type="eucl",
        const std::string outdir="") {

    int rank;
    int nprocs;
    constexpr unsigned root = 0;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Init error\n");

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // Set the num_procs
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == root) {
        printf("Init is :%s\n", init.c_str());
        if (outdir.empty())
            fprintf(stderr, "\n**[WARNING]**: No output dir specified with "
                    "'-o' flag means no output will be saved!\n");

        BOOST_LOG_TRIVIAL(info) << "Running FULL kmeans\n";
    }

    // The business
    dist_coordinator::ptr dc =
        dist_coordinator::create(
                datafn, nrow, ncol, k, max_iters, nnodes, nthread,
                rank, nprocs, p_centers, init, tolerance, dist_type);

    dc->wake4run(kpmeans::thread_state_t::ALLOC_DATA);
    dc->wait4complete();

    std::static_pointer_cast<dist_coordinator>(dc)->shift_thread_start_rid();

    struct timeval start, end;
    gettimeofday(&start , NULL);

    // Var init
    double perc_changed = std::numeric_limits<double>::max();
    bool converged = false;
    size_t iters = 0;
    size_t nchanged = 0;

    // TODO: - Many pointer casts
    //      - Many messages passed
    //      - Changing order will cut some computation e.g nchanged first ..
    //////////////////////////// Algorithm /////////////////////////////////////
    double* clstr_buff = new double[k*ncol];
    size_t* nmemb_buff = new size_t[k];

    kpmbase::clusters::ptr cltrs_ptr = std::static_pointer_cast<
        dist_coordinator>(dc)->get_gcltrs();

    // Init
    if (init == "kmeanspp")
        kmeanspp(dc, nrow, cltrs_ptr, clstr_buff, k, nprocs, rank);
    else
        dc->run_init();

    if (init == "random" || init == "forgy") {
        // MPI Update clusters
        kpmmpi::mpi::reduce_double(&(cltrs_ptr->get_means()[0]),
                clstr_buff, cltrs_ptr->size());
        cltrs_ptr->set_mean(clstr_buff);

        if (init == "random") {
            kpmmpi::mpi::reduce_size_t(&(cltrs_ptr->get_num_members_v()[0]),
                    nmemb_buff, cltrs_ptr->get_num_members_v().size());
            cltrs_ptr->set_num_members_v(nmemb_buff); // Set new counts
            cltrs_ptr->finalize_all();
            // End Init

#ifdef VERBOSE
            BOOST_VERIFY((size_t)std::accumulate(cltrs_ptr->get_num_members_v().begin(),
                        cltrs_ptr->get_num_members_v().end(), 0) == nrow);
            if (rank == root) {
                printf("New finalized centers for Proc: %d ==> \n", rank);
                cltrs_ptr->print_means();
            }
#endif
        }
    }

    // EM-step iterations
    while (iters < max_iters) {
        // Init iteration
        if (rank == root)
            printf("Running iteration %lu ...\n", iters);

        dc->wake4run(kpmeans::thread_state_t::EM);
        dc->wait4complete();
        // NOTE: Unfinalized diffs on this proc

        std::static_pointer_cast<
            dist_coordinator>(dc)->pp_aggregate();

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
        size_t pp_num_changed = dc->get_num_changed();
        kpmmpi::mpi::reduce_size_t(&pp_num_changed, &nchanged);

        if (rank == root) {
            printf("Global nchanged: %lu ...\n", nchanged);
            cltrs_ptr->print_membership_count();
        }

        BOOST_VERIFY((size_t)std::accumulate(
                    cltrs_ptr->get_num_members_v().begin(),
                    cltrs_ptr->get_num_members_v().end(), 0) == nrow);

        perc_changed = (double)nchanged/nrow; // Global perc change
        cltrs_ptr->finalize_all();

        if (nchanged == 0 || perc_changed <= tolerance) {
            converged = true;
            if (rank == root)
                printf("Algorithm converged in %lu iterations!\n", (++iters));
            break;
        }

        nchanged = 0;
        iters++;
    }

    if (!converged && rank == root)
        printf("Algorithm failed to converge in %lu iterations\n", iters);

    gettimeofday(&end, NULL);
    if (rank == root)
        printf("\nAlgorithmic time taken = %.5f sec\n",
                kpmbase::time_diff(start, end));

    if (!outdir.empty()) {
        // Collect cluster assignments
        const unsigned* local_assignments = dc->get_cluster_assignments();

        if (rank != root) {
            int rc = MPI_Ssend(local_assignments, dc->get_nrow(),
                    MPI::UNSIGNED, root, 0, MPI_COMM_WORLD);
            BOOST_ASSERT_MSG(!rc, "Failure to send local assignments to root");
        } else {
            std::vector<unsigned> assignments(nrow);
            std::copy(local_assignments, local_assignments+dc->get_nrow(),
                    assignments.begin());

            for (int pid = 1; pid < int(nprocs); pid++) {
                // Account for an uneven # of rows per process
                unsigned count;
                if (pid == (nprocs - 1))
                    count = dc->get_nrow() + (nrow % nprocs);
                else
                    count = dc->get_nrow();

                int rc = MPI_Recv(&assignments[pid*(nrow/nprocs)],
                        count, MPI::UNSIGNED, pid,
                        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                BOOST_ASSERT_MSG(!rc, "Root Failure receive local assignments");
            }

            ret = kpmbase::kmeans_t(nrow, ncol, iters, k, &assignments[0],
                    &(cltrs_ptr->get_num_members_v()[0]),
                    cltrs_ptr->get_means());

            if (!outdir.empty()) {
                printf("\nWriting output to '%s'\n", outdir.c_str());
                ret.write(outdir);
            }
        }
    }

    // MPI cleanup and graceful exit
    delete [] clstr_buff;
    delete [] nmemb_buff;
    if (p_centers) delete [] p_centers;
    MPI_Finalize();
}

static void kmeanspp(dist_coordinator::ptr dc, const size_t g_nrow,
        kpmbase::clusters::ptr cltrs_ptr, double* buff, const size_t k,
        const int nprocs, const int rank) {
    struct timeval start, end;

    std::vector<double> dist_v;
    std::vector<double> g_dist_v(g_nrow);
    dist_v.assign(dc->get_nrow(), std::numeric_limits<double>::max()); // local nrow
    dc->set_thd_dist_v_ptr(&dist_v[0]);

    // Choose c1 uniformly at random
    unsigned selected_idx = random() % g_nrow; // 0...(g_nrow-1)

    // If proc owns the row -- get it ...
    if (std::static_pointer_cast<dist_coordinator>(dc)->is_local(selected_idx)) {
        cltrs_ptr->set_mean(dc->get_thd_data(
                    std::static_pointer_cast<dist_coordinator>(dc)->local_rid(
                        selected_idx)), 0);
        dist_v[std::static_pointer_cast<dist_coordinator>(dc)->local_rid(
                selected_idx)] = 0.0;
    }

    kpmmpi::mpi::reduce_double(&(cltrs_ptr->get_means()[0]),
            buff, cltrs_ptr->size());
    cltrs_ptr->set_mean(buff);

#if VERBOSE
    BOOST_LOG_TRIVIAL(info) << "Choosing "
        << selected_idx << " as center k = 0";
#endif
    unsigned clust_idx = 0; // The number of clusters assigned

    // Choose next center c_i with weighted prob
    while ((clust_idx + 1) < k) {
        dc->set_thread_clust_idx(clust_idx); // Set the current cluster index
        dc->wake4run(KMSPP_INIT); // Run || distance comp to clust_idx
        dc->wait4complete();
        double local_cuml_dist = dc->reduction_on_cuml_sum(); // Per proc cuml dists

        double cuml_dist; // Recepticle
        kpmmpi::mpi::reduce_double(&local_cuml_dist, &cuml_dist);

        // All procs do this ...
        cuml_dist = (cuml_dist * ((double)random())) / (RAND_MAX - 1.0);
        clust_idx++;

        // Gather the g_dist_v
        kpmmpi::mpi::allgather_double(&dist_v[0],
                &g_dist_v[0], g_nrow/nprocs);

        // Gather the remaining entries from the last proc which *may* have more
        if (g_nrow % nprocs) {
            const size_t tail_idx = (g_nrow/nprocs);
            const size_t numel = (g_nrow % nprocs);

            if (rank == nprocs - 1)
                std::copy(&dist_v[tail_idx], &dist_v[tail_idx+numel],
                        &g_dist_v[g_nrow-numel]);

            kpmmpi::mpi::bcast_double(&g_dist_v[g_nrow-numel], nprocs-1, numel);
        }

        for (size_t row = 0; row < g_nrow; row++) {
            cuml_dist -= g_dist_v[row];
            if (cuml_dist <= 0) {
#if VERBOSE
                if (rank == 0)
                    BOOST_LOG_TRIVIAL(info) << "Choosing "
                        << row << " as center k = " << clust_idx;
#endif

                if (std::static_pointer_cast<dist_coordinator>(dc)->is_local(row)) {
                    cltrs_ptr->set_mean(dc->get_thd_data(
                                std::static_pointer_cast<dist_coordinator>(
                                    dc)->local_rid(row)), clust_idx);
                } else {
                    cltrs_ptr->clear();
                }
                break;
            }
        }

        kpmmpi::mpi::reduce_double(&(cltrs_ptr->get_means()[0]),
                buff, cltrs_ptr->size());
        cltrs_ptr->set_mean(buff);
        BOOST_VERIFY(cuml_dist <= 0);

    }

#if VERBOSE
    if (rank == 0) {
        BOOST_LOG_TRIVIAL(info) << "\nCluster centers after kmeans++";
        cltrs_ptr->print_means();
    }
#endif
    gettimeofday(&end, NULL);
    if (rank == 0)
        BOOST_LOG_TRIVIAL(info) << "Initialization time: " <<
            kpmbase::time_diff(start, end) << " sec\n";
}
};
} } // namespace kpmeans::dist
#endif

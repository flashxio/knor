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
#include "clusters.hpp"

namespace kpmmpi = kpmeans::mpi;

namespace kpmeans { namespace mpi {

class driver {
public:
static void run_kmeans(int argc, char* argv[],
        const std::string datafn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthread,
        const double* p_centers=NULL, const std::string init="kmeanspp",
        const double tolerance=-1, const std::string dist_type="eucl") {

    int rank;
    int nprocs;
    constexpr unsigned root = 0;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Init error\n");

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // Set the num_procs
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == root)
        BOOST_LOG_TRIVIAL(info) << "Running FULL kmeans\n";

    // The business
    kpmeans::dist_coordinator::ptr dc =
        kpmeans::dist_coordinator::create(
                datafn, nrow, ncol, k, max_iters, nnodes, nthread,
                rank, nprocs, p_centers, init, tolerance, dist_type);

    dc->wake4run(kpmeans::thread_state_t::ALLOC_DATA);
    dc->wait4complete();

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
    if (init != "random")
        throw kpmbase::not_implemented_exception();

    double* clstr_buff = new double[k*ncol];
    size_t* nmemb_buff = new size_t[k];

    // Init
    dc->run_init();
    // TODO: Check cost of all the shared_ptr passing
    kpmbase::clusters::ptr cltrs_ptr = std::static_pointer_cast<
        kpmeans::dist_coordinator>(dc)->get_gcltrs();
    // MPI Update clusters
    kpmmpi::mpi::reduce_double(&(cltrs_ptr->get_means()[0]),
            clstr_buff, cltrs_ptr->size());
    cltrs_ptr->set_mean(clstr_buff);

    kpmmpi::mpi::reduce_size_t(&(cltrs_ptr->get_num_members_v()[0]),
            nmemb_buff, cltrs_ptr->get_num_members_v().size());
    cltrs_ptr->set_num_members_v(nmemb_buff); // Set new counts
    cltrs_ptr->finalize_all();
    // End Init

    BOOST_VERIFY((size_t)std::accumulate(cltrs_ptr->get_num_members_v().begin(),
                cltrs_ptr->get_num_members_v().end(), 0) == nrow);

    // EM-step iterations
    while (iters < max_iters) {
        // Init iteration
        if (rank == root)
            printf("Running iteration %lu ...\n", iters);

        dc->wake4run(kpmeans::thread_state_t::EM);
        dc->wait4complete();

        // NOTE: Unfinalized diffs on this proc in cltrs.means
        std::static_pointer_cast<kpmeans::
            dist_coordinator>(dc)->pp_aggregate();

        // Aggregate (unfinalized) sums
        kpmmpi::mpi::reduce_double(&(cltrs_ptr->get_means()[0]),
                clstr_buff, cltrs_ptr->size());
        cltrs_ptr->set_mean(clstr_buff);

        kpmmpi::mpi::reduce_size_t(&(cltrs_ptr->get_num_members_v()[0]),
                nmemb_buff, cltrs_ptr->get_num_members_v().size());
        cltrs_ptr->set_num_members_v(nmemb_buff); // Set new counts
        cltrs_ptr->finalize_all();

        // NOTE: Now finalized
        size_t pp_num_changed = dc->get_num_changed();
        kpmmpi::mpi::reduce_size_t(&pp_num_changed, &nchanged);

        if (rank == root) {
            printf("Global nchanged: %lu ...\n", nchanged);
            cltrs_ptr->print_membership_count();
        }

#if 1
        BOOST_VERIFY((size_t)std::accumulate(
                    cltrs_ptr->get_num_members_v().begin(),
                    cltrs_ptr->get_num_members_v().end(), 0) == nrow);
#endif

        perc_changed = (double)nchanged/nrow; // Global perc change
        if (nchanged == 0 || perc_changed <= tolerance) {
            converged = true;
            if (rank == root)
                printf("Algorithm converged in %lu iterations!\n", (iters + 1));
            break;
        }
        iters++;
    }

    if (!converged && rank == root)
        printf("Algorithm failed to converge in %lu iterations\n", iters);

    gettimeofday(&end, NULL);
    if (rank == root)
        printf("\nAlgorithmic time taken = %.5f sec\n",
                kpmbase::time_diff(start, end));

    // MPI cleanup and graceful exit
    delete [] clstr_buff;
    delete [] nmemb_buff;
    if (p_centers) delete [] p_centers;
    MPI_Finalize();
}
};
} } // namespace kpmeans::mpi
#endif
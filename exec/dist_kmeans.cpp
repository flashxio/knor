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

#include <mpi.h>
#include <numa.h>
#include "dist_task_coordinator.hpp"
#include "clusters.hpp"
#include "exception.hpp"
#include "io.hpp"

#define DIST_TEST 1

static int rank;
static int nprocs;
static constexpr unsigned root = 0;

namespace kpmeans { namespace mpi {
enum tag_t {
CLUSTERS, /*For reducing clusters*/
ASSGN_CNTS, /*For reducing cluster assignment counts*/
ASSGNMTS, /*For merging assignments at end*/
};

class driver {
public:
    static void aggregate_clusters(kpmbase::prune_clusters::ptr
            proc_clusters, double* recv_buff) {
        int ret = MPI_Allreduce(&(proc_clusters->get_means()[0]), recv_buff,
                proc_clusters->size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (ret)
            throw kpmbase::mpi_exception("All reduce failure on aggregate "
                    "clusters", ret);

#if DIST_TEST
        // NOTE: We can check reduce_assignment_counts with this as well
        //  since the number of members is displayed when means are printed
        sleep(rank);
        printf("Proc: %d has per proc clusters: \n", rank);
        proc_clusters->print_means();

        // Print just to see what we get
        if (rank == root) {
            printf("Every proc agg_clust: \n");
            kpmbase::print_arr<double>(recv_buff, proc_clusters->size());
        }

#endif
        // Set new means
        proc_clusters->set_mean(recv_buff);
    }

    static void reduce_assignment_counts(kpmbase::prune_clusters::ptr
            proc_clusters, long* recv_buff) {
        int ret = MPI_Allreduce(&(proc_clusters->get_num_members_v()[0]),
                recv_buff, proc_clusters->get_num_members_v().size(), MPI::LONG,
                MPI_SUM, MPI_COMM_WORLD);
        if (ret)
            throw kpmbase::mpi_exception("All reduce failure on assingnment "
                    "counts failure", ret);
        // Set new counts
        proc_clusters->set_num_members_v(recv_buff);

        // Finalize updated clusters
        for (unsigned cl_idx = 0;
                cl_idx < proc_clusters->get_nclust(); cl_idx++)
            proc_clusters->finalize(cl_idx);
    }

    void merge_global_assignments() { /*FIXME*/ }
};
}} // namespace kpmeans::mpi

namespace kpmmpi = kpmeans::mpi;

int main(int argc, char* argv[]) {
    // Setup MPI env
    if (MPI_Init( &argc, &argv ) != MPI_SUCCESS) {
        throw std::runtime_error( "MPI_Init error\n");
    }

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // Set the num_procs
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create the pthread handler that launches multiple threads
    std::string datafn = "/data/kmeans/r10_c100_k10_rw.dat";
    size_t nrow = 100; size_t ncol = 10; size_t k = 5;
    size_t max_iters = 10; unsigned nnodes = numa_num_task_nodes();
    size_t nthread = 3; double* p_centers = NULL;
    std::string init = "random"; double tolerance = -1;
    std::string dist_type = "eucl";

    if (kpmbase::filesize(datafn.c_str()) != (sizeof(double)*nrow*ncol))
        throw kpmbase::io_exception("File size does not match input size.");

    kpmprune::dist_task_coordinator::ptr dc =
        kpmprune::dist_task_coordinator::create(
                datafn, nrow, ncol, k, max_iters, nnodes, nthread,
                rank, nprocs, p_centers, init, tolerance, dist_type);

    dc->set_global_ptrs();
    dc->wake4run(kpmeans::thread_state_t::ALLOC_DATA);
    dc->wait4complete();

    struct timeval start, end;
    gettimeofday(&start , NULL);
    //////////////////////////////////////////////////////////////
    if (init != "random")
        throw kpmbase::not_implemented_exception();

    double* clstr_buff = new double[k*ncol];
    long* nmemb_buff = new long[k];

    dc->run_init();

    // TODO: Check cost of all the shared_ptr passing
    kpmbase::prune_clusters::ptr cls_ptrs = std::static_pointer_cast<
        kpmprune::dist_task_coordinator>(dc)->get_gcltrs();
    kpmmpi::driver::aggregate_clusters(cls_ptrs, clstr_buff);
    kpmmpi::driver::reduce_assignment_counts(cls_ptrs, nmemb_buff);

#if DIST_TEST
    BOOST_VERIFY((size_t)std::accumulate(cls_ptrs->get_num_members_v().begin(),
                cls_ptrs->get_num_members_v().end(), 0) == nrow);

    if (rank == root) {
        printf("New finalized centers for Proc: %d ==> \n", rank);
        cls_ptrs->print_means();
    }
#endif
    //////////////////////////////////////////////////////////////

    if (rank == root)
        BOOST_LOG_TRIVIAL(info) << "\n\nAlgorithmic time taken = " <<
            kpmbase::time_diff(start, end) << " sec\n";

    // MPI cleanup and graceful exit
    delete [] clstr_buff;
    delete [] nmemb_buff;
    MPI_Finalize();
    return EXIT_SUCCESS;
}

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

static int rank;
static int nprocs;

namespace kpmeans { namespace mpi {

enum tag_t {
CLUSTERS, /*For reducing clusters*/
ASSGN_CNTS, /*For reducing cluster assignment counts*/
ASSGNMTS, /*For merging assignments at end*/
};

class driver {

public:
    static void aggregate_clusters(kpmbase::prune_clusters::ptr
            , const bool prune) {
    }

    void reduce_assignment_counts() {}
    void merge_global_assignments() {}
};
}} // namespace kpmeans::mpi

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
#if 1
    kpmprune::dist_task_coordinator::ptr dc =
        kpmprune::dist_task_coordinator::create(
                datafn, nrow, ncol, k, max_iters, nnodes, nthread,
                rank, nprocs, p_centers, init, tolerance, dist_type);
#else

#include "kmeans_task_coordinator.hpp"
    kpmprune::kmeans_task_coordinator::ptr dc =
        kpmprune::kmeans_task_coordinator::create(
                datafn, nrow, ncol, k, max_iters, nnodes, nthread, p_centers,
                init, tolerance, dist_type);
#endif

    dc->set_global_ptrs();
    dc->wake4run(kpmeans::thread_state_t::ALLOC_DATA);
    dc->wait4complete();

    //////////////////////////////////////////////////////////////
    sleep(rank);
    dc->print_thread_data();
    //////////////////////////////////////////////////////////////

    // MPI cleanup and graceful exit
    MPI_Finalize();
    return EXIT_SUCCESS;
}

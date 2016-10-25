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

#include <numa.h>

#include "signal.h"
#include "dist_task_coordinator.hpp"
#include "clusters.hpp"
#include "exception.hpp"
#include "io.hpp"
#include "thread_state.hpp"
#include "mpi.hpp"
#include "util.hpp"

#define DIST_TEST 1

static int rank;
static int nprocs;
static constexpr unsigned root = 0;

namespace kpmmpi = kpmeans::mpi;

static void print_usage();

int main(int argc, char* argv[]) {
    // Setup MPI env
    if (MPI_Init( &argc, &argv) != MPI_SUCCESS) {
        throw std::runtime_error( "MPI_Init error\n");
    }

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // Set the num_procs
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if 0
    // Create the pthread handler that launches multiple threads
    std::string datafn = "/data/kmeans/r10_c100_k10_rw.dat";
    size_t nrow = 100; size_t ncol = 10; size_t k = 5;
    size_t max_iters = 10; unsigned nnodes = numa_num_task_nodes();
    size_t nthread = 3; double* p_centers = NULL;
    std::string init = "random"; double tolerance = -1;
    std::string dist_type = "eucl";
#endif

    if (argc < 5) {
        print_usage();
        exit(EXIT_FAILURE);
    }

	int opt;
    std::string datafn = std::string(argv[1]);
    size_t nrow = atol(argv[2]);
    size_t ncol = atol(argv[3]);
    unsigned k = atol(argv[4]);

    std::string dist_type = "eucl";
    std::string centersfn = "";
	unsigned max_iters=std::numeric_limits<unsigned>::max();
	std::string init = "kmeanspp";
	unsigned nthread = kpmbase::get_num_omp_threads();
	int num_opts = 0;
	double tolerance = -1;
    bool use_min_tri = false;
    unsigned nnodes = numa_num_task_nodes();

    // Increase by 3 -- getopt ignores argv[0]
	argv += 3;
	argc -= 3;

	signal(SIGINT, kpmbase::int_handler);
	while ((opt = getopt(argc, argv, "l:i:t:T:d:C:mN:")) != -1) {
		num_opts++;
		switch (opt) {
			case 'l':
				tolerance = atof(optarg);
				num_opts++;
				break;
			case 'i':
				max_iters = atol(optarg);
				num_opts++;
				break;
			case 't':
				init = optarg;
				num_opts++;
				break;
			case 'T':
				nthread = atoi(optarg);
				num_opts++;
				break;
			case 'd':
				dist_type = std::string(optarg);
				num_opts++;
				break;
			case 'C':
				centersfn = std::string(optarg);
                BOOST_ASSERT_MSG(kpmbase::is_file_exist(centersfn.c_str()),
                        "Centers file name doesn't exit!");
                init = "none"; // Ignore whatever you pass in
				num_opts++;
				break;
			case 'm':
				use_min_tri = true;
				num_opts++;
				break;
			case 'N':
				nnodes = atoi(optarg);
				num_opts++;
				break;
			default:
				print_usage();
		}
	}

    BOOST_ASSERT_MSG(!(init=="none" && centersfn.empty()),
            "Centers file name doesn't exit!");

    if (kpmbase::filesize(datafn.c_str()) != (sizeof(double)*nrow*ncol))
        throw kpmbase::io_exception("File size does not match input size.");

    double* p_centers = NULL;

    if (kpmbase::is_file_exist(centersfn.c_str())) {
        p_centers = new double [k*ncol];
        kpmbase::bin_reader<double> br2(centersfn, k, ncol);
        br2.read(p_centers);
        printf("Read centers!\n");
    }

    // The business
    kpmprune::dist_task_coordinator::ptr dc =
        kpmprune::dist_task_coordinator::create(
                datafn, nrow, ncol, k, max_iters, nnodes, nthread,
                rank, nprocs, p_centers, init, tolerance, dist_type);

    dc->set_global_ptrs();
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
    kpmbase::prune_clusters::ptr cltrs_ptr = std::static_pointer_cast<
        kpmprune::dist_task_coordinator>(dc)->get_gcltrs();
    // MPI Update clusters
    kpmmpi::mpi::reduce_double(&(cltrs_ptr->get_means()[0]),
            clstr_buff, cltrs_ptr->size());
    cltrs_ptr->set_mean(clstr_buff);

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

    // EM-step iterations
    while (iters < max_iters) {
        if (iters == 1)
            std::static_pointer_cast<kpmprune::
                dist_task_coordinator>(dc)->set_prune_init(false);

        // Init iteration
        if (rank == root)
            printf("Running iteration %lu ...\n", iters);

        std::static_pointer_cast<kpmprune::
            dist_task_coordinator>(dc)->get_dm()->compute_dist(cltrs_ptr, ncol);
#ifdef VERBOSE
        if (rank == 0) {
            printf("Updated dist matrix:\n");
            std::static_pointer_cast<kpmprune::
             dist_task_coordinator>(dc)->get_dm()->print();
        }
#endif
        dc->wake4run(kpmeans::thread_state_t::EM);
        dc->wait4complete();
        // NOTE: Unfinalized diffs on this proc in cltrs.means
        std::static_pointer_cast<kpmprune::
            dist_task_coordinator>(dc)->pp_aggregate();

        // NOTE: cltrs_ptr has this procs diff (agg of threads from this proc)
        // NOTE: clstr_buff has agg of all procs diff
        kpmmpi::mpi::reduce_double(&(cltrs_ptr->get_means()[0]),
                clstr_buff, cltrs_ptr->size());

        // nmemb_buff has agg of all procs diff on membership count
        kpmmpi::mpi::reduce_size_t(&(cltrs_ptr->get_num_members_v()[0]),
                nmemb_buff, cltrs_ptr->get_num_members_v().size());

        if (iters == 0) {
            cltrs_ptr->set_mean(clstr_buff);
            cltrs_ptr->set_num_members_v(nmemb_buff);
        } else {
            // Get the prev univ clusters
            cltrs_ptr->set_mean(cltrs_ptr->get_prev_means());
            cltrs_ptr->set_num_members_v(&(std::static_pointer_cast<kpmprune::
                        dist_task_coordinator>(dc)->get_prev_num_members())[0]);
#ifdef VERBOSE
            printf("Prev universal clusters for Proc: %d ==> \n", rank);
            cltrs_ptr->print_means();
#endif
            cltrs_ptr->set_complete_all(); // Must set this
            cltrs_ptr->unfinalize_all();

            cltrs_ptr->means_peq(clstr_buff);
            cltrs_ptr->num_members_v_peq(nmemb_buff);
        }

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
        if (nchanged == 0 || perc_changed <= tolerance) {
            converged = true;
            if (rank == root)
                printf("Algorithm converged in %lu iterations!\n", (iters + 1));
            break;
        }

        for (unsigned c = 0; c < k; c++) {
            cltrs_ptr->finalize(c);
            cltrs_ptr->set_prev_dist(
                    kpmbase::eucl_dist(&(cltrs_ptr->get_means()[c*ncol]),
                        &(cltrs_ptr->get_prev_means()[c*ncol]), ncol), c);
#ifdef VERBOSE
            BOOST_LOG_TRIVIAL(info) << "Dist to prev mean for c:" << c
                << " is " << cltrs_ptr->get_prev_dist(c);
#endif
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
    return EXIT_SUCCESS;
}

void print_usage() {
	fprintf(stderr,
            "mpirun.mpich -n NUM_PROCS dist_kmeans data-file num-rows"
            " num-cols k [alg-options]\n");
    fprintf(stderr, "-t type: type of initialization for kmeans"
           " ['random', 'forgy', 'kmeanspp', 'none']\n");
    fprintf(stderr, "-T num_thread: The number of threads per process\n");
    fprintf(stderr, "-i iters: maximum number of iterations\n");
    fprintf(stderr, "-C File with initial clusters in same format as data\n");
    fprintf(stderr, "-l tolerance for convergence (1E-6)\n");
    fprintf(stderr, "-d Distance metric [eucl,cos]\n");
    fprintf(stderr, "-m Use the minimal triangle inequality (~Elkan's alg)\n");
    fprintf(stderr, "-N No. of numa nodes you want to use\n");
}

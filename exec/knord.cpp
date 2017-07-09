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
#include <getopt.h>

#include "signal.h"

#include "dist_task_coordinator.hpp"
#include "dist_coordinator.hpp"
#include "io.hpp"
#include "util.hpp"

static int rank;
static int nprocs;
static constexpr unsigned root = 0;

static void print_usage();

int main(int argc, char* argv[]) {
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
    bool no_prune = false;
    unsigned nnodes = kpmbase::get_num_nodes();
    std::string outdir = "";

    // Increase by 3 -- getopt ignores argv[0]
	argv += 3;
	argc -= 3;

	while ((opt = getopt(argc, argv, "l:i:t:T:d:C:PN:o:")) != -1) {
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
                kpmbase::assert_msg(kpmbase::is_file_exist(centersfn.c_str()),
                        "Centers file name doesn't exit!");
                init = "none"; // Ignore whatever you pass in
				num_opts++;
				break;
			case 'P':
				no_prune = true;
				num_opts++;
				break;
			case 'N':
				nnodes = atoi(optarg);
				num_opts++;
				break;
			case 'o':
				outdir = std::string(optarg);
				num_opts++;
				break;
			default:
				print_usage();
                exit(EXIT_FAILURE);
		}
	}

    kpmbase::assert_msg(!(init=="none" && centersfn.empty()),
            "Centers file name doesn't exit!");

    if (kpmbase::filesize(datafn.c_str()) != (sizeof(double)*nrow*ncol))
        throw kpmbase::io_exception("File size does not match input size.");

    double* p_centers = NULL;

    if (kpmbase::is_file_exist(centersfn.c_str())) {
        p_centers = new double [k*ncol];
        kpmbase::bin_io<double> br2(centersfn, k, ncol);
        br2.read(p_centers);
        printf("Read centers!\n");
    }

    kpmbase::kmeans_t ret; // Only root fills this

    if (no_prune) {
        kpmeans::dist::dist_coordinator::ptr dc =
            kpmeans::dist::dist_coordinator::create(argc, argv,
                    datafn, nrow, ncol, k, max_iters, nnodes, nthread,
                    p_centers, init, tolerance, dist_type);
        std::static_pointer_cast<kpmeans::dist::dist_coordinator>(
                dc)->run_kmeans(ret, outdir);
    } else {
        kpmeans::prune::dist_task_coordinator::ptr dc =
            kpmeans::prune::dist_task_coordinator::create(argc, argv,
                    datafn, nrow, ncol, k, max_iters, nnodes, nthread,
                    p_centers, init, tolerance, dist_type);
        std::static_pointer_cast<kpmeans::prune::dist_task_coordinator>(
                dc)->run_kmeans(ret, outdir);
    }

    if (p_centers) delete [] p_centers;
    return EXIT_SUCCESS;
}

void print_usage() {
	fprintf(stderr,
            "mpirun.mpich -n NUM_PROCS knord data-file nsamples"
            " dim k [alg-options]\n");
    fprintf(stderr, "-t type: type of initialization for kmeans"
           " ['random', 'forgy', 'kmeanspp', 'none']\n");
    fprintf(stderr, "-T num_thread: The number of threads per process\n");
    fprintf(stderr, "-i iters: maximum number of iterations\n");
    fprintf(stderr, "-C File with initial clusters in same format as data\n");
    fprintf(stderr, "-l tolerance for convergence (1E-6)\n");
    fprintf(stderr, "-d Distance metric [eucl,cos]\n");
    fprintf(stderr, "-P DO NOT use the minimal triangle inequality (~Elkan's alg)\n");
    fprintf(stderr, "-N No. of numa nodes you want to use\n");
    fprintf(stderr, "-o Write output to an output directory of this name\n");
}

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


#include <limits>
#ifdef USE_NUMA
#include <numa.h>
#endif

#include <getopt.h>

#include "signal.h"
#include "io.hpp"
#ifdef __linux
#include "kmeans.hpp"
#endif

#include "kmeans_coordinator.hpp"
#include "kmeans_task_coordinator.hpp"
#include "util.hpp"

static void print_usage();
namespace kpmbase = kpmeans::base;
namespace kpmprune = kpmeans::prune;

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
	size_t max_iters=std::numeric_limits<size_t>::max();
	std::string init = "kmeanspp";
	unsigned nthread = kpmbase::get_num_omp_threads();
	int num_opts = 0;
	double tolerance = -1;
    bool no_prune = false;
    bool omp = false;
    if (omp) { }
    unsigned nnodes = kpmbase::get_num_nodes();
    std::string outdir = "";

    // Increase by 3 -- getopt ignores argv[0]
	argv += 3;
	argc -= 3;

	while ((opt = getopt(argc, argv, "l:i:t:T:d:C:PON:o:")) != -1) {
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
			case 'O':
				omp = true;
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
		}
	}

    if (outdir.empty())
        fprintf(stderr, "\n\n**[WARNING]**: No output dir specified with '-o' "
                " flag means no output will be saved!\n\n");

    kpmbase::assert_msg(!(init == "none" && centersfn.empty()),
            "Centers file name doesn't exit!");

    if (kpmbase::filesize(datafn.c_str()) != (sizeof(double)*nrow*ncol))
        throw kpmbase::io_exception("File size does not match input size.");

    double* p_centers = NULL;
    kpmbase::kmeans_t ret;

    if (kpmbase::is_file_exist(centersfn.c_str())) {
        p_centers = new double [k*ncol];
        kpmbase::bin_io<double> br2(centersfn, k, ncol);
        br2.read(p_centers);
        printf("Read centers!\n");
    } else
        printf("No centers to read ..\n");
#ifdef __linux
    if (omp) {
        kpmbase::bin_io<double> br(datafn, nrow, ncol);
        double* p_data = new double [nrow*ncol];
        br.read(p_data);
        printf("Read data!\n");

        unsigned* p_clust_asgns = new unsigned [nrow];
        size_t* p_clust_asgn_cnt = new size_t [k];

        if (NULL == p_centers) // We have no preallocated centers
            p_centers = new double [k*ncol];

        if (no_prune) {
            ret = kpmeans::omp::compute_kmeans(p_data, p_centers, p_clust_asgns,
                    p_clust_asgn_cnt, nrow, ncol, k, max_iters,
                    nthread, init, tolerance, dist_type);
        } else {
            ret = kpmeans::omp::compute_min_kmeans(p_data, p_centers, p_clust_asgns,
                    p_clust_asgn_cnt, nrow, ncol, k, max_iters,
                    nthread, init, tolerance, dist_type);
        }

        delete [] p_clust_asgns;
        delete [] p_clust_asgn_cnt;
        delete [] p_data;
    } else {
#endif
        if (no_prune) {
            kpmeans::kmeans_coordinator::ptr kc =
                kpmeans::kmeans_coordinator::create(datafn,
                    nrow, ncol, k, max_iters, nnodes, nthread, p_centers,
                    init, tolerance, dist_type);
            ret = kc->run_kmeans();
        } else {
            kpmprune::kmeans_task_coordinator::ptr kc =
                kpmprune::kmeans_task_coordinator::create(
                    datafn, nrow, ncol, k, max_iters, nnodes, nthread, p_centers,
                    init, tolerance, dist_type);
            ret = kc->run_kmeans();
        }
#ifdef __linux
    }
#endif

    if (!outdir.empty()) {
        printf("\nWriting output to '%s'\n", outdir.c_str());
        ret.write(outdir);
    }

    if (p_centers) delete [] p_centers;

    return EXIT_SUCCESS;
}

void print_usage() {
	fprintf(stderr,
        "knori data-file nsamples dim k [alg-options]\n");
    fprintf(stderr, "-t type: type of initialization for kmeans"
           " ['random', 'forgy', 'kmeanspp', 'none']\n");
    fprintf(stderr, "-T num_thread: The number of threads to run\n");
    fprintf(stderr, "-i iters: maximum number of iterations\n");
    fprintf(stderr, "-C File with initial clusters in same format as data\n");
    fprintf(stderr, "-l tolerance for convergence (1E-6)\n");
    fprintf(stderr, "-d Distance metric [eucl,cos]\n");
    fprintf(stderr, "-P DO NOT use the minimal triangle inequality (~Elkan's alg)\n");
    fprintf(stderr, "-O Use OpenMP for ||ization rather than fast pthreads\n");
    fprintf(stderr, "-N No. of numa nodes you want to use\n");
    fprintf(stderr, "-o Write output to an output directory of this name\n");
    exit(EXIT_FAILURE);
}

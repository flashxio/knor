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
#ifdef __unix__
#include "kmeans.hpp"
#endif

#include "kmeans_coordinator.hpp"
#include "kmeans_task_coordinator.hpp"
#include "util.hpp"

#include "cxxopts/cxxopts.hpp"

namespace kpmbase = kpmeans::base;
namespace kpmprune = kpmeans::prune;

int main(int argc, char* argv[])
{
  try {
    // positional args
    std::string datafn = "";
    unsigned k = 0;

    // optional args
    unsigned nthread = kpmbase::get_num_omp_threads();
    std::string dist_type = "eucl";
    std::string centersfn = "";
    unsigned max_iters=std::numeric_limits<unsigned>::max();
    std::string init = "kmeanspp";
    double tolerance = -1;

    bool no_prune = false;
    bool omp = false;

    if (omp) { }
    unsigned nnodes = kpmbase::get_num_nodes();
    std::string outdir = "";

    cxxopts::Options options(argv[0],
            "knori data-file nsamples dim k [alg-options]\n");
    options.positional_help("[optional args]");

    options.add_options()
      ("f,datafn", "Path to data-file on disk",
            cxxopts::value<std::string>(datafn), "FILE")
      ("n,nsamples", "Number of samples in the dataset (rows)",
            cxxopts::value<std::string>())
      ("m,dim", "Number of features in the dataset (columns)",
            cxxopts::value<std::string>())
      ("k,nclust", "Number of clusters desired",
            cxxopts::value<unsigned>(k))
      ("T,num_thread", "The number of threads to run",
            cxxopts::value<unsigned>(nthread))
      ("i,iters", "maximum number of iterations",
            cxxopts::value<unsigned>(max_iters))
      ("C,centersfn", "Path to centroids on disk",
            cxxopts::value<std::string>(centersfn), "FILE")
      ("O,omp", "Use OpenMP for ||ization rather than fast pthreads",
            cxxopts::value<bool>(omp))
      ("P,prune", "DO NOT use the minimal triangle inequality (~Elkan's alg)",
            cxxopts::value<bool>(no_prune))
      ("N,nnodes", "No. of numa nodes you want to use",
            cxxopts::value<unsigned>(nnodes))
      ("d,dist", "Distance metric [eucl,cos]",
            cxxopts::value<std::string>(dist_type))
      ("l,tol", "tolerance for convergence (1E-6)",
            cxxopts::value<std::string>())
      ("o,outdir", "Write output to an output directory of this name",
            cxxopts::value<std::string>(outdir))
      ("h,help", "Print help")
    ;

    options.parse_positional({"datafn", "nsamples", "dim", "nclust"});
    int nargs = argc;
    options.parse(argc, argv);

    if (options.count("help") || (nargs == 1)) {
        std::cout << options.help() << std::endl;
        exit(EXIT_SUCCESS);
    }

    if (nargs < 5) {
        std::cout << "[ERROR]: Not enough default arguments\n";
        std::cout << options.help() << std::endl;
        exit(EXIT_SUCCESS);
    }

    kpmbase::assert_msg(kpmbase::is_file_exist(datafn.c_str()),
            "Data file name doesn't exit!");
    size_t nrow = atol(options["nsamples"].as<std::string>().c_str());
    size_t ncol = atol(options["dim"].as<std::string>().c_str());
    if (options.count("tol"))
        tolerance = std::stod(options["tol"].as<std::string>());
    if (options.count("centersfn")) {
        kpmbase::assert_msg(kpmbase::is_file_exist(centersfn.c_str()),
                "Centers file name doesn't exit!");
        init = "none";  // Ignore whatever you pass in
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
#ifdef __unix__
    if (omp) {
        kpmbase::bin_io<double> br(datafn, nrow, ncol);
        double* p_data = new double [nrow*ncol];
        br.read(p_data);
        printf("Read data!\n");

        unsigned* p_clust_asgns = new unsigned [nrow];
        size_t* p_clust_asgn_cnt = new size_t [k];

        if (NULL == p_centers)  // We have no preallocated centers
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
#ifdef __unix__
    }
#endif

    if (!outdir.empty()) {
        printf("\nWriting output to '%s'\n", outdir.c_str());
        ret.write(outdir);
    }

    if (p_centers) delete [] p_centers;

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
    return EXIT_SUCCESS;
}

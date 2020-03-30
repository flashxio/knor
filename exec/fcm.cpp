/*
 * Copyright 2016 neurodata (http://neurodata.io/)
 * Written by Disa Mhembere (disa@jhu.edu)
 *
 * This file is part of knor
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

#include <getopt.h>

#include "signal.h"
#include "io.hpp"

#include "fcm_coordinator.hpp"
#include "cxxopts/cxxopts.hpp"

namespace clustercore = knor::core;

int main(int argc, char* argv[]) {
  try {
    // positional args
    std::string datafn = "";
    unsigned k = 0;

    // optional args
    unsigned nthread = clustercore::get_num_omp_threads();
    std::string dist_type = "sqeucl";
    std::string centersfn = "";
    unsigned max_iters=std::numeric_limits<unsigned>::max();
    std::string init = "forgy";
    double tolerance = 1E-6;
    unsigned fuzzindex = 2;

    unsigned nnodes = clustercore::get_num_nodes();
    std::string outdir = "";

    cxxopts::Options options(argv[0],
            "fcm data-file nsamples dim k [alg-options]\n");
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
      ("z,fuzzindex", "fuzziness index [1,inf)",
            cxxopts::value<unsigned>(fuzzindex))
      ("C,centersfn", "Path to centroids on disk",
            cxxopts::value<std::string>(centersfn), "FILE")
      ("N,nnodes", "No. of numa nodes you want to use",
            cxxopts::value<unsigned>(nnodes))
      ("d,dist", "Distance metric [eucl,cos, taxi]",
            cxxopts::value<std::string>(dist_type))
      ("l,tol", "tolerance for convergence (1E-6)",
            cxxopts::value<std::string>())
      ("o,outdir", "Write output to an output directory of this name",
            cxxopts::value<std::string>(outdir))
      ("h,help", "Print help");

    options.parse_positional({"datafn", "nsamples", "dim",
            "nclust"});
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

    clustercore::assert_msg(clustercore::is_file_exist(datafn.c_str()),
            "Data file name doesn't exit!");
    size_t nrow = atol(options["nsamples"].as<std::string>().c_str());
    size_t ncol = atol(options["dim"].as<std::string>().c_str());
    if (options.count("tol"))
        tolerance = std::stod(options["tol"].as<std::string>());
    if (options.count("centersfn")) {
        clustercore::assert_msg(clustercore::is_file_exist(centersfn.c_str()),
                "Centers file name doesn't exit!");
        init = "none";  // Ignore whatever you pass in
    }

    if (outdir.empty())
        fprintf(stderr, "\n\n**[WARNING]**: No output dir specified with '-o' "
                " flag means no output will be saved!\n\n");

    clustercore::assert_msg(!(init == "none" && centersfn.empty()),
            "Centers file name doesn't exit!");

    if (clustercore::filesize(datafn.c_str()) != (sizeof(double)*nrow*ncol))
        throw clustercore::io_exception("File size does not match input size.");

    std::vector<double> centers;

    if (clustercore::is_file_exist(centersfn.c_str())) {
        centers.resize(k*ncol);
        clustercore::bin_io<double> br2(centersfn, k, ncol);
        br2.read(&centers[0]);
        printf("Read centers!\n");
    } else {
        printf("No centers to read ..\n");
    }

    auto ret = knor::fcm_coordinator::create(datafn, nrow, ncol, k,
                max_iters, nnodes, nthread, &centers[0],
                init, tolerance, dist_type, fuzzindex)->run();

    if (!outdir.empty()) {
        printf("\nWriting output to '%s'\n", outdir.c_str());
        ret.write(outdir);
    }

  } catch (const cxxopts::OptionException& e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
    return EXIT_SUCCESS;
}

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


#include <limits>
#ifdef USE_NUMA
#include <numa.h>
#endif

#include <getopt.h>

#include "signal.h"
#include "io.hpp"

#include "binding/kmeanspp.hpp"
#include "util.hpp"

#include "cxxopts/cxxopts.hpp"

namespace kbase = knor::base;

int main(int argc, char* argv[]) {
  try {
    // positional args
    std::string datafn = "";
    unsigned k = 0;

    // optional args
    unsigned nthread = kbase::get_num_omp_threads();
    std::string dist_type = "eucl";
    unsigned nstarts = 10;

    unsigned nnodes = kbase::get_num_nodes();
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
      ("s,num_starts", "maximum number of iterations",
            cxxopts::value<unsigned>(nstarts))
      ("N,nnodes", "No. of numa nodes you want to use",
            cxxopts::value<unsigned>(nnodes))
      ("d,dist", "Distance metric [eucl,cos]",
            cxxopts::value<std::string>(dist_type))
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

    if (nargs < 4) {
        std::cout << "[ERROR]: Not enough default arguments\n";
        std::cout << options.help() << std::endl;
        exit(EXIT_SUCCESS);
    }

    kbase::assert_msg(kbase::is_file_exist(datafn.c_str()),
            "Data file name doesn't exit!");
    size_t nrow = atol(options["nsamples"].as<std::string>().c_str());
    size_t ncol = atol(options["dim"].as<std::string>().c_str());

    if (outdir.empty())
        fprintf(stderr, "\n\n**[WARNING]**: No output dir specified with '-o' "
                " flag means no output will be saved!\n\n");

    if (kbase::filesize(datafn.c_str()) != (sizeof(double)*nrow*ncol))
        throw kbase::io_exception("File size does not match input size.");


    auto ret = kbase::kmeansPP(datafn, nrow, ncol, k, nstarts,
            nthread, dist_type);
    printf("Best start: %u, Best energy: %f\n", ret.first.first, ret.first.second);
    printf("Best Clustering: \n") ;
    kbase::print(ret.second.assignment_count);

    if (!outdir.empty()) {
        printf("\nWriting output to '%s'\n", outdir.c_str());
        ret.second.write(outdir);
    }

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
    return EXIT_SUCCESS;
}

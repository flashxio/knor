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

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_NUMA
#include <numa.h>
#endif

#include <atomic>
#include <fstream>
#include <thread>
#include <cassert>

#include "util.hpp"
#include "exception.hpp"
#include <cmath>

namespace knor { namespace base {
double get_bic(const std::vector<double>& dist_v, const size_t nrow,
        const size_t ncol, const unsigned k) {
        double bic = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:bic) shared (dist_v)
#endif
    for (unsigned i = 0; i < dist_v.size(); i++) {
        bic += (dist_v[i] );
    }
#ifndef BIND
    printf("Distance sum: %f\n", bic);
#endif

    return 2*bic + std::log(static_cast<double>(nrow))*ncol*k;
}

void spherical_projection(double* data, const size_t nrow,
        const size_t ncol) {
#ifdef _OPENMP
#pragma omp parallel for shared (data)
#endif
    for (unsigned row = 0; row < nrow; row++) {
        double norm2 = 0;
        for (unsigned col = 0; col < ncol; col++)
            norm2 += (data[row]*data[row]);
        for (unsigned col = 0; col < ncol; col++)
            data[col] = data[col]/std::sqrt(norm2);
    }
}

unsigned get_hclust_ceil(const unsigned k) {
    return std::pow(2, static_cast<unsigned>(std::ceil(std::log2(k))));
}

unsigned get_hclust_floor(const unsigned k) {
    return std::pow(2, static_cast<unsigned>(std::log2(k)));
}

// Verbatim from FlashX
float time_diff(struct timeval time1, struct timeval time2) {
    return time2.tv_sec - time1.tv_sec +
        ((float)(time2.tv_usec - time1.tv_usec))/1000000;
}

int get_num_omp_threads() {
    int num_threads = std::thread::hardware_concurrency();
    if (!num_threads) {
#ifndef BIND
        std::cout << "\n\n[WARNING]: Failed to detect # of CPUs/threads!" <<
            " Using default: 1\n\n";
#endif
        num_threads = 1;
    }
    return num_threads;
}

init_t get_init_type(const std::string init) {
    if (init == "random")
        return init_t::RANDOM;
    else if (init == "forgy")
        return init_t::FORGY;
    else if (init == "kmeanspp")
        return init_t::PLUSPLUS;
    else if (init == "none")
        return init_t::NONE;
    else
        throw thread_exception(std::string("param init must be one of:"
                    " [random | forgy | kmeanspp]. It is '")
                + init + std::string("'"));
}

dist_t get_dist_type(const std::string dist_type) {
    if (dist_type == "eucl")
        return dist_t::EUCL;
    else if (dist_type == "cos")
        return dist_t::COS;
    else if (dist_type == "taxi")
        return dist_t::TAXI;
    else if (dist_type == "sqeucl")
        return dist_t::SQEUCL;
    else
        throw thread_exception(std::string
                ("[ERROR]: param dist_type must be one of: 'eucl', 'cos', "
                 "'taxi', 'sqeucl'. It is '") + dist_type + std::string("'"));
}

bool is_file_exist(const char *fn) {
    std::ifstream infile(fn);
    return infile.good();
}

size_t filesize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate
            | std::ifstream::binary);
    return in.tellg();
}


unsigned get_num_nodes() {
#ifdef USE_NUMA
    return static_cast<unsigned>(numa_num_task_nodes());
#else
    return 1;
#endif
}

void assert_msg(bool expr, const std::string msg) {
    if (!expr) {
#ifndef BIND
        std::cerr << msg << std::endl;
#endif
        assert(0);
    }
}
} } // End namespace knor::base

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

#ifdef LINUX
#include <omp.h>
#endif

#include <atomic>
#include <fstream>
#include <thread>

#include "util.hpp"
#include "exception.hpp"

namespace kpmeans { namespace base {
double get_bic(const std::vector<double>& dist_v, const size_t nrow,
        const size_t ncol, const unsigned k) {
        double bic = 0;

#ifdef LINUX
#pragma omp parallel for reduction(+:bic) shared (dist_v)
#endif
    for (unsigned i = 0; i < dist_v.size(); i++) {
        bic += (dist_v[i] );
    }
#ifndef BIND
    printf("Distance sum: %f\n", bic);
#endif

    return 2*bic + log(nrow)*ncol*k;
}

void spherical_projection(double* data, const size_t nrow,
        const size_t ncol) {
#ifdef LINUX
#pragma omp parallel for shared (data)
#endif
    for (unsigned row = 0; row < nrow; row++) {
        double norm2 = 0;
        for (unsigned col = 0; col < ncol; col++)
            norm2 += (data[row]*data[row]);
        for (unsigned col = 0; col < ncol; col++)
            data[col] = data[col]/sqrt(norm2);
    }
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

init_type_t get_init_type(const std::string init) {
    if (init == "random")
        return init_type_t::RANDOM;
    else if (init == "forgy")
        return init_type_t::FORGY;
    else if (init == "kmeanspp")
        return init_type_t::PLUSPLUS;
    else if (init == "none")
        return init_type_t::NONE;
    else
        throw thread_exception(std::string("param init must be one of:"
                    " [random | forgy | kmeanspp]. It is '")
                + init + std::string("'"));
}

dist_type_t get_dist_type(const std::string dist_type) {
    if (dist_type == "eucl")
        return dist_type_t::EUCL;
    else if (dist_type == "cos")
        return dist_type_t::COS;
    else
        throw thread_exception(std::string
                ("[ERROR]: param dist_type must be one of: 'eucl', 'cos'."
                 " It is '") + dist_type + std::string("'"));
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
} }

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

#ifndef __KNOR_SKMEANS_HPP__
#define __KNOR_SKMEANS_HPP__

#include "skmeans_coordinator.hpp"


namespace knor { namespace base {
    // NOTE: It is the callers job to allocate/free data & p_centers

cluster_t skmeans(double* data, const size_t nrow,
        const size_t ncol, const unsigned k,
        size_t max_iters=std::numeric_limits<size_t>::max(),
        unsigned nnodes=get_num_nodes(),
        unsigned nthread=get_num_omp_threads(),
        double* p_centers=NULL, std::string init="kmeanspp",
        double tolerance=-1) {

    if (p_centers)
        init = "none";

    knor::coordinator::ptr kc =
        knor::skmeans_coordinator::create("",
                nrow, ncol, k, max_iters, nnodes, nthread, p_centers,
                init, tolerance, "cos");
    return kc->run(data);
}

cluster_t skmeans(std::string datafn, const size_t nrow,
        const size_t ncol, const unsigned k,
        size_t max_iters=std::numeric_limits<size_t>::max(),
        unsigned nnodes=get_num_nodes(),
        unsigned nthread=get_num_omp_threads(),
        double* p_centers=NULL, std::string init="kmeanspp",
        double tolerance=-1) {

    knor::coordinator::ptr kc =
        knor::skmeans_coordinator::create(datafn,
                nrow, ncol, k, max_iters, nnodes, nthread, p_centers,
                init, tolerance, "cos");

    return kc->run();
}
} } // End namespace knor::base
#endif

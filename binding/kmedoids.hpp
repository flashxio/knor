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

#ifndef __KNOR_BIND_KMEDOIDS_HPP__
#define __KNOR_BIND_KMEDOIDS_HPP__

#include "medoid_coordinator.hpp"


namespace knor { namespace base {
    // NOTE: It is the callers job to allocate/free data & p_centers

cluster_t kmedoids(double* data, const size_t nrow,
        const size_t ncol, const unsigned k,
        size_t max_iters=std::numeric_limits<size_t>::max(),
        unsigned nnodes=get_num_nodes(),
        unsigned nthread=get_num_omp_threads(),
        double* p_centers=NULL, std::string init="forgy",
        double tolerance=-1, std::string dist_type="taxi") {

    if (p_centers)
        init = "none";

    knor::medoid_coordinator::ptr kc =
        knor::medoid_coordinator::create("",
                nrow, ncol, k, max_iters, nnodes, nthread, p_centers,
                init, tolerance, dist_type);
    return kc->run(data);
}

cluster_t kmedoids(std::string datafn, const size_t nrow,
        const size_t ncol, const unsigned k,
        size_t max_iters=std::numeric_limits<size_t>::max(),
        unsigned nnodes=get_num_nodes(),
        unsigned nthread=get_num_omp_threads(),
        double* p_centers=NULL, std::string init="forgy",
        double tolerance=-1, std::string dist_type="taxi") {

    std::vector<double> data(nrow*ncol);
    kbase::bin_io<double> br(datafn, nrow, ncol);
    br.read(&data);
    return kmedoids(&data[0], nrow, ncol, k, max_iters, nnodes,
            nthread, p_centers, init, tolerance, dist_type);
}
} } // End namespace knor::base
#endif

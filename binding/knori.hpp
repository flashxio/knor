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

#ifndef __KNORI_HPP__
#define __KNORI_HPP__

#include <limits>
#include <numa.h>

#include "signal.h"
#include "io.hpp"
#include "kmeans.hpp"

#include "kmeans_coordinator.hpp"
#include "kmeans_task_coordinator.hpp"
#include "util.hpp"

namespace kpmeans { namespace base {
kpmbase::kmeans_t kmeans(const std::string datafn, const size_t nrow,
        const size_t ncol, const unsigned k,
        size_t max_iters=std::numeric_limits<size_t>::max(),
        unsigned nnodes=numa_num_task_nodes(),
        unsigned nthread=kpmbase::get_num_omp_threads(),
        double* p_centers=NULL, std::string init="kmeanspp",
        double tolerance=-1, std::string dist_type="eucl",
        std::string centersfn = "", bool omp=false) {

    kpmbase::kmeans_t ret;

    if (kpmbase::is_file_exist(centersfn.c_str())) {
        p_centers = new double [k*ncol];
        kpmbase::bin_io<double> br2(centersfn, k, ncol);
        br2.read(p_centers);
        printf("Read centers!\n");
    } else
        printf("No centers to read ..\n");


    if (omp) {
        kpmeans::kmeans_coordinator::ptr kc =
            kpmeans::kmeans_coordinator::create(datafn,
                    nrow, ncol, k, max_iters, nnodes, nthread, p_centers,
                    init, tolerance, dist_type);
        ret = kc->run_kmeans();
    } else {
        kpmeans::kmeans_coordinator::ptr kc =
            kpmeans::kmeans_coordinator::create(datafn,
                    nrow, ncol, k, max_iters, nnodes, nthread, p_centers,
                    init, tolerance, dist_type);
        ret = kc->run_kmeans();
    }

    if (p_centers) { delete [] p_centers; }
    ret.print();
    return ret;
}

} } // End namespace kpmeans::base
#endif

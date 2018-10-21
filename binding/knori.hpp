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

#ifndef __KNOR_BIND_KNORI_HPP__
#define __KNOR_BIND_KNORI_HPP__

#include "io.hpp"
#include "../libauto/kmeans.hpp"

#include "kmeans_coordinator.hpp"
#include "kmeans_task_coordinator.hpp"
#include "util.hpp"

#ifdef USE_NUMA
#include "numa_reorg.hpp"
namespace kbind = knor::binding;
#endif

#ifdef _OPENMP
namespace komp = knor::omp;
#endif

namespace kprune = knor::prune;

namespace knor { namespace base {

// NOTE: It is the callers job to allocate/free data & p_centers
coordinator::ptr _kmeans(const size_t nrow,
        const size_t ncol, const unsigned k,
        size_t max_iters=std::numeric_limits<size_t>::max(),
        unsigned nnodes=get_num_nodes(),
        unsigned nthread=get_num_omp_threads(),
        double* p_centers=NULL, std::string init="kmeanspp",
        double tolerance=-1, std::string dist_type="eucl") {

    if (p_centers)
        init = "none";

    return kprune::kmeans_task_coordinator::create(
            "", nrow, ncol, k, max_iters, nnodes,
            nthread, p_centers,
            init, tolerance, dist_type);
}

coordinator::ptr _kmeans(const std::string datafn, const size_t nrow,
        const size_t ncol, const unsigned k,
        size_t max_iters=std::numeric_limits<size_t>::max(),
        unsigned nnodes=get_num_nodes(),
        unsigned nthread=get_num_omp_threads(),
        double* p_centers=NULL, std::string init="kmeanspp",
        double tolerance=-1, std::string dist_type="eucl") {

    if (p_centers)
        init = "none";

    return kprune::kmeans_task_coordinator::create(
            datafn, nrow, ncol, k, max_iters, nnodes, nthread, p_centers,
            init, tolerance, dist_type);
}


// NOTE: It is the callers job to allocate/free data & p_centers
cluster_t kmeans(double* data, const size_t nrow,
        const size_t ncol, const unsigned k,
        size_t max_iters=std::numeric_limits<size_t>::max(),
        unsigned nnodes=get_num_nodes(),
        unsigned nthread=get_num_omp_threads(),
        double* p_centers=NULL, std::string init="kmeanspp",
        double tolerance=-1, std::string dist_type="eucl") {

    return _kmeans(nrow, ncol, k, max_iters, nnodes, nthread,
            p_centers, init, tolerance, dist_type)->run(data);
}

cluster_t kmeans(const std::string datafn, const size_t nrow,
        const size_t ncol, const unsigned k,
        size_t max_iters=std::numeric_limits<size_t>::max(),
        unsigned nnodes=get_num_nodes(),
        unsigned nthread=get_num_omp_threads(),
        double* p_centers=NULL, std::string init="kmeanspp",
        double tolerance=-1, std::string dist_type="eucl") {

    return _kmeans(datafn, nrow, ncol, k, max_iters, nnodes, nthread,
            p_centers, init, tolerance, dist_type)->run();
}

std::pair<std::pair<unsigned, double>, cluster_t> kmeansPP(
        double* data, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned nstart=1,
        unsigned nthread=get_num_omp_threads(), std::string dist_type="eucl") {

    auto nnodes = get_num_nodes();
    unsigned best_start = 1;
    auto coord = std::static_pointer_cast<kprune::kmeans_task_coordinator>(
            _kmeans(nrow, ncol, k, 0, nnodes, nthread,
                NULL, "kmeanspp", -1, dist_type));

    cluster_t best_cluster_t = coord->run(data);
    coord->tally_assignment_counts();
    double best_energy = coord->compute_cluster_energy();

    for (unsigned start = 1; start < nstart; start++) {
        coord->reinit();
        coord->tally_assignment_counts();
        auto energy = coord->compute_cluster_energy();

        if (energy < best_energy) {
            best_cluster_t = coord->dump_state();
            best_energy = energy;
            best_start = start + 1;
        }
    }

    auto _ = std::pair<unsigned, double>(best_start, best_energy);
    return std::pair<std::pair<unsigned, double>, cluster_t>(_, best_cluster_t);
}

std::pair<std::pair<unsigned, double>, cluster_t> kmeansPP(
        std::string datafn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned nstart=1,
        unsigned nthread=get_num_omp_threads(), std::string dist_type="eucl") {

    auto nnodes = get_num_nodes();
    unsigned best_start = 1;
    auto coord = std::static_pointer_cast<kprune::kmeans_task_coordinator>(
            _kmeans(datafn, nrow, ncol, k, 0, nnodes, nthread,
                NULL, "kmeanspp", -1, dist_type));

    cluster_t best_cluster_t = coord->run();
    coord->tally_assignment_counts();
    double best_energy = coord->compute_cluster_energy();

    for (unsigned start = 1; start < nstart; start++) {
        coord->reinit();
        coord->tally_assignment_counts();
        auto energy = coord->compute_cluster_energy();

        if (energy < best_energy) {
            best_cluster_t = coord->dump_state();
            best_energy = energy;
            best_start = start + 1;
        }
    }

    auto _ = std::pair<unsigned, double>(best_start, best_energy);
    return std::pair<std::pair<unsigned, double>, cluster_t>(_, best_cluster_t);
}

} } // End namespace knor::base
#endif

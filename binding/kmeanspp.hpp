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

#ifndef __KNOR_BIND_KMEANSPP_HPP__
#define __KNOR_BIND_KMEANSPP_HPP__

#include "kmeans_task_coordinator.hpp"

namespace kprune = knor::prune;

namespace knor { namespace base {

typedef std::pair<std::pair<unsigned, double>, cluster_t> pp_pair;

pp_pair kmeansPP(
        double* data, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned nstart=1,
        unsigned nthread=get_num_omp_threads(), std::string dist_type="eucl") {

    auto nnodes = get_num_nodes();
    unsigned best_start = 1;

    auto coord = std::static_pointer_cast<kprune::kmeans_task_coordinator>(
            kprune::kmeans_task_coordinator::create(
            "", nrow, ncol, k, 0, nnodes, nthread, NULL,
            "kmeanspp", -1, dist_type));

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
    return pp_pair(_, best_cluster_t);
}

pp_pair kmeansPP(
        std::string datafn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned nstart=1,
        unsigned nthread=get_num_omp_threads(), std::string dist_type="eucl") {

    auto nnodes = get_num_nodes();
    unsigned best_start = 1;
    auto coord = std::static_pointer_cast<kprune::kmeans_task_coordinator>(
            kprune::kmeans_task_coordinator::create(
            datafn, nrow, ncol, k, 0, nnodes, nthread,
                NULL, "kmeanspp", -1, dist_type));

    cluster_t best_cluster_t = coord->run();
    coord->tally_assignment_counts();
    double best_energy = coord->compute_cluster_energy();

    struct timeval start, end;
    gettimeofday(&start , NULL);

    for (unsigned start = 1; start < nstart; start++) {
#ifndef BIND
        printf("start: %u ...\n", start);
#endif
        coord->reinit();
        coord->tally_assignment_counts();
        auto energy = coord->compute_cluster_energy();

        if (energy < best_energy) {
            best_cluster_t = coord->dump_state();
            best_energy = energy;
            best_start = start + 1;
        }
    }

    gettimeofday(&end, NULL);
#ifndef BIND
    printf("\n\nAlgorithmic time taken = %.6f sec\n",
        base::time_diff(start, end));
    printf("\n******************************************\n");
#endif

    auto _ = std::pair<unsigned, double>(best_start, best_energy);
    return pp_pair(_, best_cluster_t);
}
} } // End namespace knor::base
#endif

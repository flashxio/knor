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
#ifndef __KNOR_GMEANS_COORDINATOR_HPP__
#define __KNOR_GMEANS_COORDINATOR_HPP__

#include "xmeans_coordinator.hpp"

namespace knor {

class gmeans_coordinator : public xmeans_coordinator {
    private:
        const short strictness;
        std::vector<double> cluster_diff_v;
        std::vector<double> cluster_diff_scalar;

    public:
        gmeans_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned kmax, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const base::init_t it,
                const double tolerance, const base::dist_t dt,
                const unsigned min_clust_size, const short strictness);

        typedef std::shared_ptr<gmeans_coordinator> ptr;

        static xmeans_coordinator::ptr create(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers=NULL, const std::string init="forgy",
                const double tolerance=-1, const std::string dist_type="eucl",
                const unsigned min_clust_size=2, const short strictness=4) {

            base::init_t _init_t = base::get_init_type(init);
            base::dist_t _dist_t = base::get_dist_type(dist_type);
#if KM_TEST
#ifndef BIND
            printf("gmeans coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "min_clust_size: %u\n\n", nnodes, nthreads, nrow, ncol,
                    init.c_str(), dist_type.c_str(), fn.c_str(), min_clust_size);
#endif
#endif
            return xmeans_coordinator::ptr(
                    new gmeans_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t, tolerance, _dist_t,
                    min_clust_size, strictness));
        }

        void build_thread_state() override;
        // Pass file handle to threads to read & numa alloc
        virtual base::cluster_t run(double* allocd_data=NULL,
            const bool numa_opt=false) override;
        void partition_decision() override;
        void compute_cluster_diffs();
        void assemble_ad_vecs(std::unordered_map<unsigned,
                std::vector<double>>& ad_vecs);
        void compute_ad_stats(
                std::unordered_map<unsigned, std::vector<double>>& ad_vecs);
        void deactivate(const unsigned id) override;
        void activate(const unsigned id) override;
};
}
#endif

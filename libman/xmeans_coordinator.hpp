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
#ifndef __KNOR_XMEANS_COORDINATOR_HPP__
#define __KNOR_XMEANS_COORDINATOR_HPP__

#include "hclust_coordinator.hpp"

#if 1
#include <unordered_map>
#endif

namespace knor {

struct split_score_t {

    unsigned pid; // parent
    unsigned lid; // left child
    unsigned rid; // right child
    double pscore; // Parent score
    double cscore; // Child score

    split_score_t(const unsigned pid, const unsigned lid, const unsigned rid) :
        pid(pid), lid(lid), rid(rid),
        pscore(std::numeric_limits<double>::min()),
        cscore(std::numeric_limits<double>::min()) {
        }

    split_score_t() :
        pscore(std::numeric_limits<double>::min()),
        cscore(std::numeric_limits<double>::min()) {
        }
};

class xmeans_coordinator : public hclust_coordinator {
    public:
        xmeans_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const base::init_t it,
                const double tolerance, const base::dist_t dt,
                const unsigned min_clust_size);

        typedef std::shared_ptr<xmeans_coordinator> ptr;

        std::vector<double> partition_dist; // Data point to partition dist
        std::vector<double> nearest_cdist; // Data point to centroid dist
        std::shared_ptr<base::clusters> cltrs; // Record partition -> data point
        bool compute_pdist; // Should threads comp the partition_dist this iter?

        static hclust_coordinator::ptr create(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers=NULL, const std::string init="forgy",
                const double tolerance=-1, const std::string dist_type="eucl",
                const unsigned min_clust_size=2) {

            base::init_t _init_t = base::get_init_type(init);
            base::dist_t _dist_t = base::get_dist_type(dist_type);
#if KM_TEST
#ifndef BIND
            printf("xmeans coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "min_clust_size: %u\n\n", nnodes, nthreads, nrow, ncol,
                    init.c_str(), dist_type.c_str(), fn.c_str(), min_clust_size);
#endif
#endif
            return hclust_coordinator::ptr(
                    new xmeans_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t, tolerance, _dist_t,
                    min_clust_size));
        }

        template <typename T>
        void accumulate(const std::vector<T>& in,
                std::unordered_map<T, std::vector<T>>& out) {
            for (size_t i = 0; i < in.size(); i++) {
                out[in[i]].push_back(i);
            }
        }
        virtual void build_thread_state() override;
        // Pass file handle to threads to read & numa alloc
        virtual base::cluster_t run(double* allocd_data=NULL,
            const bool numa_opt=false) override;
        virtual void combine_partition_means();
        virtual void partition_decision();
        void bic(split_score_t& score,
                std::unordered_map<unsigned, std::vector<unsigned>>& memb_cltrs);

        void compute_bic_scores(
                std::vector<split_score_t>& bic_scores,
                std::unordered_map<unsigned,
                std::vector<unsigned>>& memb_cltrs);
};
}
#endif

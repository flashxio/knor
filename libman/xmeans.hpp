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

#ifndef __KNOR_XMEANS_HPP__
#define __KNOR_XMEANS_HPP__

#include "hclust.hpp"

namespace knor {
class xmeans : public hclust {
    protected:
        std::vector<double>& partition_dist; // Data point to partition dist
        std::vector<double>& nearest_cdist; // Data point to centroid dist
        const bool& compute_pdist;
        std::shared_ptr<kbase::clusters> g_clusters;

        xmeans(const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol, unsigned k,
                hclust_map& g_hcltrs,
                unsigned* cluster_assignments,
                const std::string fn, base::dist_t dist_metric,
                const std::shared_ptr<base::thd_safe_bool_vector> cltr_active_vec,
                std::vector<double>& partition_dist,
                std::vector<double>& nearest_cdist,
                const bool& compute_pdist);
    public:
        static hclust::ptr create(
                const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol, unsigned k,
                hclust_map& g_hcltrs,
                unsigned* cluster_assignments, const std::string fn,
                base::dist_t dist_metric,
                const std::shared_ptr<base::thd_safe_bool_vector> cltr_active_vec,
                std::vector<double>& partition_dist,
                std::vector<double>& nearest_cdist,
                const bool& compute_pdist) {
            return hclust::ptr(
                        new xmeans(node_id, thd_id, start_rid,
                        nprocrows, ncol, k, g_hcltrs,
                        cluster_assignments, fn, dist_metric,
                        cltr_active_vec, partition_dist, nearest_cdist,
                        compute_pdist));
        }

        virtual void start(const thread_state_t state) override;
        // Given the current ID split it into two (or not)
        virtual void H_EM_step() override; // Similar to EM step
        void set_g_clusters(std::shared_ptr<kbase::clusters> g_clusters) {
            this->g_clusters = g_clusters;
        }
        virtual void partition_mean() override;
        virtual ~xmeans() {}
};
} // End namespace knor
#endif

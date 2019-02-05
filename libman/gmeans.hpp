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

#ifndef __KNOR_GMEANS_HPP__
#define __KNOR_GMEANS_HPP__

#include "xmeans.hpp"

namespace knor {
class gmeans : public xmeans {
    using xmeans::xmeans;

    public:
        static xmeans::ptr create(
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
            return xmeans::ptr(
                        new gmeans(node_id, thd_id, start_rid,
                        nprocrows, ncol, k, g_hcltrs,
                        cluster_assignments, fn, dist_metric,
                        cltr_active_vec, partition_dist, nearest_cdist,
                        compute_pdist));
        }

        void start(const thread_state_t state) override;
        void H_split_step() override;
};
} // End namespace knor
#endif

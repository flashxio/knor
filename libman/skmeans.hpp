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

#include <vector>
#include "kmeans_thread.hpp"

namespace knor { namespace base {
    class clusters;
} }

namespace kbase = knor::base;

namespace knor {
class skmeans : public kmeans_thread  {
    private:
        using kmeans_thread::kmeans_thread;

        std::vector<double> min_feature_val; // min
        std::vector<double> max_feature_val; // max

        skmeans(const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol,
                std::shared_ptr<kbase::clusters> g_clusters,
                unsigned* cluster_assignments,
                const std::string fn, kbase::dist_t dist_metric);
    public:
        static thread::ptr create(
                const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol,
                std::shared_ptr<kbase::clusters> g_clusters,
                unsigned* cluster_assignments, const std::string fn,
                const kbase::dist_t dist_metric) {
            return thread::ptr(
                    new skmeans(node_id, thd_id, start_rid,
                        nprocrows, ncol, g_clusters,
                        cluster_assignments, fn, dist_metric));
        }

        void feature_bounds_reduction(); // Min & Max of all features
        void feature_normalize();
        void run() override;

        std::vector<double>& get_min_feature_val() {
            return min_feature_val;
        }

        std::vector<double>& get_max_feature_val() {
            return max_feature_val;
        }
};
}
#endif

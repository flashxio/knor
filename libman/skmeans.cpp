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

#include "skmeans.hpp"

#include "kmeans_thread.hpp"
#include "clusters.hpp"

namespace knor {
    skmeans::skmeans(const int node_id, const unsigned thd_id,
            const unsigned start_rid, const unsigned nprocrows,
            const unsigned ncol,
            std::shared_ptr<kbase::clusters> g_clusters,
            unsigned* cluster_assignments,
            const std::string fn, kbase::dist_t dist_metric) :
        kmeans_thread(node_id, thd_id, start_rid,
                nprocrows, ncol, g_clusters,
                cluster_assignments, fn, dist_metric) {

            min_feature_val.assign(ncol, std::numeric_limits<double>::max());
            max_feature_val.assign(ncol, std::numeric_limits<double>::min());
        }

    // Min & Max of all features
    void skmeans::feature_bounds_reduction() {
        for (unsigned row = 0; row < nprocrows; row++) {
            for (unsigned col = 0; col < ncol; col++) {
                if (local_data[row*ncol+col] < min_feature_val[col])
                    min_feature_val[col] = local_data[row*ncol+col];
                if (local_data[row*ncol+col] > max_feature_val[col])
                    max_feature_val[col] = local_data[row*ncol+col];
            }
        }
    }

    // Min-Max normilization
    void skmeans::feature_normalize() {
       // NOTE: Only use AFTER feature_bounds_reduction AND coordinator reduce
        for (unsigned row = 0; row < nprocrows; row++) {
            for (unsigned col = 0; col < ncol; col++) {
                local_data[row*ncol+col] =
                    (local_data[row*ncol+col] - min_feature_val[col])/
                    (max_feature_val[col] - min_feature_val[col]);
            }
        }
    }

    void skmeans::run() {
        switch(state) {
            case ALLOC_DATA:
                numa_alloc_mem();
                break;
            case KMSPP_INIT:
                kmspp_dist();
                break;
            case EM: /*E step of kmeans*/
                EM_step();
                break;
            case BOUNDS:
                feature_bounds_reduction();
                break;
            case NORMALIZE_DATA:
                feature_normalize();
                break;
            case EXIT:
                throw kbase::thread_exception(
                        "Thread state is EXIT but running!\n");
            default:
                throw kbase::thread_exception("Unknown thread state\n");
        }
        sleep();
    }
}
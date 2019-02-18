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

#include <iostream>
#include <cassert>

#include "xmeans.hpp"
#include "types.hpp"
#include "util.hpp"
#include "io.hpp"
#include "clusters.hpp"
#include "thd_safe_bool_vector.hpp"

namespace knor {

    xmeans::xmeans(const int node_id, const unsigned thd_id,
            const unsigned start_rid, const unsigned nprocrows,
            const unsigned ncol, unsigned k,
            hclust_map& g_hcltrs,
            unsigned* cluster_assignments,
            const std::string fn, base::dist_t dist_metric,
            const base::thd_safe_bool_vector::ptr cltr_active_vec,
            std::vector<double>& partition_dist,
            std::vector<double>& nearest_cdist, const bool& compute_pdist) :
        hclust(node_id, thd_id, start_rid, nprocrows, ncol, k,
                g_hcltrs, cluster_assignments, fn, dist_metric,
                cltr_active_vec), partition_dist(partition_dist),
                nearest_cdist(nearest_cdist), compute_pdist(compute_pdist) {

            local_clusters = kbase::sparse_clusters::create(
                                base::get_max_hnodes(k*2), ncol);
            // Use this for the mean of the full partition calculation
        }

void xmeans::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback<xmeans>, this);
    if (rc)
        throw base::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

void xmeans::partition_mean() {
    local_clusters->clear(); // This sets all means to 0 -- no resizing

    for (unsigned row = 0; row < nprocrows; row++) {
        unsigned true_row_id = get_global_data_id(row);

        // Not active
        if (!cltr_active_vec->get(cluster_assignments[true_row_id]))
            continue; // Skip it

        auto rpart_id = part_id[true_row_id];
        local_clusters->add_member(&local_data[row*ncol], rpart_id);
    }
}


void xmeans::H_EM_step() {
    local_hcltrs.clear();

    nchanged.clear();

    auto itr = g_hcltrs.get_iterator();
    while (itr.has_next()) {
        auto kv = itr.next();
        if (kv.second->has_converged()) {
            continue;
        }

        // No need to set id, zeroid, oneid because global hcltrs knows them
        local_hcltrs[kv.first] = base::h_clusters::create(2, ncol);
        local_hcltrs[kv.first]->clear(); // NOTE: Could be combined into ctor
    }

    for (unsigned row = 0; row < nprocrows; row++) {
        // What cluster is this row in?
        unsigned true_row_id = get_global_data_id(row);
        auto rpart_id = part_id[true_row_id];

        // Get distance to partition
        if (compute_pdist) {
            partition_dist[true_row_id] =
                base::dist_comp_raw<double>(&local_data[row*ncol],
                    &(g_clusters->
                        get_means()[rpart_id*ncol]), ncol, dist_metric);
        }

        // Not active
        if (!cltr_active_vec->get(cluster_assignments[true_row_id]) ||
                g_hcltrs[rpart_id]->has_converged())
            continue; // Skip it

        unsigned asgnd_clust = base::INVALID_CLUSTER_ID;
        bool flag = 0; // Is the best the zeroid or oneid?
        double best, dist;
        dist = best = std::numeric_limits<double>::max();

        for (unsigned clust_idx = 0; clust_idx < 2; clust_idx++) {
            dist = base::dist_comp_raw<double>(&local_data[row*ncol],
                    &(g_hcltrs[rpart_id]->
                        get_means()[clust_idx*ncol]), ncol, dist_metric);
            if (dist < best) {
                best = dist;
                if (clust_idx == 0) {
                    asgnd_clust = g_hcltrs[rpart_id]->get_zeroid();
                } else {
                    asgnd_clust = g_hcltrs[rpart_id]->get_oneid();
                    flag = 1;
                }
            }
        }

        nearest_cdist[true_row_id] = best; // Update the best dist
        assert(asgnd_clust != base::INVALID_CLUSTER_ID);

        if (asgnd_clust != cluster_assignments[true_row_id])
            nchanged[rpart_id]++;
        local_hcltrs[rpart_id]->add_member(&local_data[row*ncol], flag);
        cluster_assignments[true_row_id] = asgnd_clust;
    }
}
} // End namespace knor


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

#include "kmeans_thread.hpp"
#include "types.hpp"
#include "util.hpp"
#include "io.hpp"
#include "clusters.hpp"

namespace knor {
kmeans_thread::kmeans_thread(const int node_id, const unsigned thd_id,
        const unsigned start_rid,
        const unsigned nprocrows, const unsigned ncol,
        kbase::clusters::ptr g_clusters, unsigned* cluster_assignments,
        const std::string fn, kbase::dist_t dist_metric) :
            thread(node_id, thd_id, ncol,
            cluster_assignments, start_rid, fn, dist_metric),
        g_clusters(g_clusters), nprocrows(nprocrows){

            local_clusters =
                kbase::clusters::create(g_clusters->get_nclust(), ncol);
            set_data_size(sizeof(double)*nprocrows*ncol);
        }

void kmeans_thread::run() {
    switch(state) {
        case TEST:
            test();
            break;
        case ALLOC_DATA:
            numa_alloc_mem();
            break;
        case KMSPP_INIT:
            kmspp_dist();
            break;
        case EM: /*E step of kmeans*/
            EM_step();
            break;
        case EXIT:
            throw kbase::thread_exception(
                    "Thread state is EXIT but running!\n");
        default:
            throw kbase::thread_exception("Unknown thread state\n");
    }
    sleep();
}

void kmeans_thread::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback<kmeans_thread>, this);
    if (rc)
        throw kbase::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

void kmeans_thread::EM_step() {
    meta.num_changed = 0; // Always reset at the beginning of an EM-step
    local_clusters->clear();

    for (unsigned row = 0; row < nprocrows; row++) {
        unsigned asgnd_clust = kbase::INVALID_CLUSTER_ID;
        double best, dist;
        dist = best = std::numeric_limits<double>::max();

        for (unsigned clust_idx = 0;
                clust_idx < g_clusters->get_nclust(); clust_idx++) {
            dist = kbase::dist_comp_raw<double>(&local_data[row*ncol],
                    &(g_clusters->get_means()[clust_idx*ncol]), ncol,
                    dist_metric);

            if (dist < best) {
                best = dist;
                asgnd_clust = clust_idx;
            }
        }

        assert(asgnd_clust != kbase::INVALID_CLUSTER_ID);
        unsigned true_row_id = get_global_data_id(row);

        if (asgnd_clust != cluster_assignments[true_row_id])
            meta.num_changed++;

        cluster_assignments[true_row_id] = asgnd_clust;
        local_clusters->add_member(&local_data[row*ncol], asgnd_clust);
    }
}

/** Method for a distance computation vs a single cluster.
 * Used in kmeans++ init
 */
void kmeans_thread::kmspp_dist() {
    unsigned clust_idx = meta.clust_idx;
    for (unsigned row = 0; row < nprocrows; row++) {
        unsigned true_row_id = get_global_data_id(row);

        double dist = kbase::dist_comp_raw<double>(&local_data[row*ncol],
                &((g_clusters->get_means())[clust_idx*ncol]), ncol,
                dist_metric);

        if (dist < dist_v[true_row_id]) { // Found a closer cluster than before
            dist_v[true_row_id] = dist;
            cluster_assignments[true_row_id] = clust_idx;
        }
        cuml_dist += dist_v[true_row_id];
    }
}

} // End namespace knor

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

#include "hclust.hpp"
#include "types.hpp"
#include "util.hpp"
#include "io.hpp"
#include "clusters.hpp"
#include "hclust_id_generator.hpp"

namespace {
void* callback(void* arg) {
    knor::hclust* t = static_cast<knor::hclust*>(arg);
#ifdef USE_NUMA
    t->bind2node_id();
#endif

    while (true) { // So we can receive task after task
        if (t->get_state() == knor::WAIT)
            t->wait();

        if (t->get_state() == knor::EXIT) {// No more work to do
            break;
        }
        t->run(); // else
    }

    // We've stopped running so exit
    pthread_exit(NULL);

#ifdef _WIN32
    return NULL;
#endif
}
}

namespace knor {
hclust::hclust(const int node_id, const unsigned thd_id,
        const unsigned start_rid,
        const unsigned nprocrows, const unsigned ncol,
        hclust_map* g_hcltrs, unsigned* cluster_assignments,
        const std::string fn, base::dist_t dist_metric) :
            thread(node_id, thd_id, ncol,
            cluster_assignments, start_rid, fn, dist_metric) {

            local_clusters = nullptr; // Not used here
            this->nprocrows = nprocrows;
            this->g_hcltrs = g_hcltrs; // Global clusters
            this->k = g_hcltrs->at(0)->get_nclust();

            nchanged[0] = 0;

            set_data_size(sizeof(double)*nprocrows*ncol);
        }

void hclust::run() {
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
        case H_EM:
            H_EM_step();
            break;
        case H_SPLIT:
            H_split_step();
            break;
        case EXIT:
            throw base::thread_exception(
                    "Thread state is EXIT but running!\n");
        default:
            throw base::thread_exception("Unknown thread state\n");
    }
    sleep();
}

void hclust::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback, this);
    if (rc)
        throw base::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

// Simply pick a new partition
void hclust::H_split_step() {
    for (unsigned row = 0; row < nprocrows; row++) {
        unsigned true_row_id = get_global_data_id(row);
        part_id[true_row_id] = cluster_assignments[true_row_id];
    }
}

void hclust::H_EM_step() {

    base::reset(nchanged);

    local_hcltrs.clear();
    for (auto kv : (*g_hcltrs))
        local_hcltrs[kv.first] = base::h_clusters::create(2, ncol);

    for (unsigned row = 0; row < nprocrows; row++) {
        // What cluster is this row in?
        unsigned true_row_id = get_global_data_id(row);
        auto curr_clust = cluster_assignments[true_row_id];

        // Not active
        // TODO: Deal with deactivating clusters ...
        if (!((*cltr_active_vec)[curr_clust]))
            continue; // Skip it

        auto rpart_id = part_id[true_row_id];
        unsigned asgnd_clust = base::INVALID_CLUSTER_ID;
        double best, dist;
        dist = best = std::numeric_limits<double>::max();

        for (unsigned clust_idx = 0; clust_idx < 2; clust_idx++) {
            dist = base::dist_comp_raw<double>(&local_data[row*ncol],
                    &(g_hcltrs->at(rpart_id)->
                        get_means()[clust_idx*ncol]), ncol, dist_metric);

            std::cout << "Diff btwn data: \n";
            base::print_arr<double>(&local_data[row*ncol], ncol);
            std::cout << "Centers: \n";
            base::print_arr<double>(
                    &(g_hcltrs->at(rpart_id)->get_means()[clust_idx*ncol]), ncol);

            std::cout << "rid: " << true_row_id << ", dist: " << dist <<
                ", to cid: " << clust_idx << std::endl;

            if (dist < best) {
                best = dist;
                if (clust_idx == 0) {
                    asgnd_clust = g_hcltrs->at(rpart_id)->get_zeroid();
                } else {
                    asgnd_clust = g_hcltrs->at(rpart_id)->get_oneid();
                }
            }
        }

        assert(asgnd_clust != base::INVALID_CLUSTER_ID);
        std::cout << "ROW: " << true_row_id << ", assigned to: " << asgnd_clust
            << "\n";

        if (asgnd_clust != cluster_assignments[true_row_id])
            ; // TODO: meta.num_changed++;

        cluster_assignments[true_row_id] = asgnd_clust;
        local_hcltrs[rpart_id]->add_member(&local_data[row*ncol], asgnd_clust);
    }
}

/** Method for a distance computation vs a single cluster.
 * Used in kmeans++ init
 */
void hclust::kmspp_dist() {
#if 0
    unsigned clust_idx = meta.clust_idx;
    for (unsigned row = 0; row < nprocrows; row++) {
        unsigned true_row_id = get_global_data_id(row);

        double dist = base::dist_comp_raw<double>(&local_data[row*ncol],
                &((g_hcltrs->get_means())[clust_idx*ncol]), ncol,
                dist_metric);

        if (dist < dist_v[true_row_id]) { // Found a closer cluster than before
            dist_v[true_row_id] = dist;
            cluster_assignments[true_row_id] = clust_idx;
        }
        cuml_dist += dist_v[true_row_id];
    }
#endif
}
} // End namespace knor
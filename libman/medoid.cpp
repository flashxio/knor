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

#include "medoid.hpp"
#include "types.hpp"
#include "util.hpp"
#include "io.hpp"
#include "clusters.hpp"
#include "medoid_coordinator.hpp"

namespace knor {
medoid::medoid(const int node_id, const unsigned thd_id,
        const unsigned start_rid,
        const unsigned nprocrows, const unsigned ncol,
        kbase::clusters::ptr g_clusters, unsigned* cluster_assignments,
        const std::string fn, const double sample_rate):
            thread(node_id, thd_id, ncol, cluster_assignments, start_rid, fn),
            g_clusters(g_clusters), nprocrows(nprocrows),
            sample_rate(sample_rate) {

            local_clusters =
                kbase::clusters::create(g_clusters->get_nclust(), ncol);
            set_data_size(sizeof(double)*nprocrows*ncol);
            local_medoid_energy.assign(g_clusters->get_nclust(),0);

            // For sampling
            ur_distribution = std::uniform_real_distribution<double>(0.0, 1.0);
        }

void medoid::run() {
    switch(state) {
        case TEST:
            test();
            break;
        case ALLOC_DATA:
            numa_alloc_mem();
            break;
        case EM:
            EM_step();
            break;
        case MEDOID:
            medoid_step();
            break;
        case EXIT:
            throw kbase::thread_exception(
                    "Thread state is EXIT but running!\n");
        default:
            throw kbase::thread_exception("Unknown thread state\n");
    }
    sleep();
}

void medoid::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback<medoid>, this);
    if (rc)
        throw kbase::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

void medoid::EM_step() {
    meta.num_changed = 0; // Always reset at the beginning of an EM-step
    local_clusters->clear();
    local_medoid_energy.assign(g_clusters->get_nclust(), 0);

    for (unsigned row = 0; row < nprocrows; row++) {
        // Choose row as new cluster center
        unsigned asgnd_clust = kbase::INVALID_CLUSTER_ID;
        double best, dist;
        dist = best = std::numeric_limits<double>::max();
        unsigned true_rid = get_global_data_id(row);

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

        if (asgnd_clust != cluster_assignments[true_rid])
            meta.num_changed++;

        // Add to the cluster energy
        local_medoid_energy[asgnd_clust] += best;

        cluster_assignments[true_rid] = asgnd_clust;

        // We only need the cluster membership count
        // We don't update the global cltrs with these numbers
        local_clusters->num_members_peq(1, asgnd_clust);
    }
}

void medoid::medoid_step() {
    // Reset
    candidate_medoids.assign(g_clusters->get_nclust(), -1);
    candidate_medoid_energy.assign(g_clusters->get_nclust(),
            std::numeric_limits<double>::max());

    // TODO: Choose them then batch them by cid
    for (unsigned row = 0; row < nprocrows; row++) {
        // Sample a few cluster members
        if (ur_distribution(generator) > sample_rate)
            continue;

        unsigned true_rid = get_global_data_id(row);
        // What cluster the row is in
        unsigned cid = cluster_assignments[true_rid];
        double energy = 0;

        // member_id is a global identifier
        for (auto const& member_id : coord->get_membership()[cid]) {
            if (member_id != true_rid) {
                energy += kbase::dist_comp_raw<double>(&local_data[row*ncol],
                    coord->get_thd_data(member_id), ncol, dist_metric);
            }
        }

        // We have a possible better choice of medoid
        if (energy < candidate_medoid_energy[cid]) {
            candidate_medoid_energy[cid] = energy;
            candidate_medoids[cid] = true_rid;
        }
    }
}

} // End namespace knor

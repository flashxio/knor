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

#include "medoid_thread.hpp"
#include "types.hpp"
#include "util.hpp"
#include "io.hpp"
#include "clusters.hpp"
#include "dist_matrix.hpp"

namespace knor {
medoid_thread::medoid_thread(const int node_id, const unsigned thd_id,
        const unsigned start_rid,
        const unsigned nprocrows, const unsigned ncol,
        kbase::clusters::ptr g_clusters, unsigned* cluster_assignments,
        const std::string fn, std::shared_ptr<kprune::dist_matrix> pw_dm,
        double* global_medoid_energy):
            thread(node_id, thd_id, ncol, cluster_assignments, start_rid, fn) {

            this->nprocrows = nprocrows;
            this->g_clusters = g_clusters;
            this->pw_dm = pw_dm;
            this->global_medoid_energy = global_medoid_energy;

            local_clusters =
                kbase::clusters::create(g_clusters->get_nclust(), ncol);
            set_data_size(sizeof(double)*nprocrows*ncol);
            local_medoid_energy.assign(g_clusters->get_nclust(),0);
#if VERBOSE
#ifndef
            std::cout << "Initializing thread. Metadata: thd_id: "
                << this->thd_id << ", start_rid: " << this->start_rid <<
                ", node_id: " << this->node_id << ", nprocrows: " <<
                this->nprocrows << ", ncol: " << this->ncol << std::endl;
#endif
#endif
        }

void medoid_thread::sleep() {
    int rc;
    rc = pthread_mutex_lock(&mutex);
    if (rc) perror("pthread_mutex_lock");

    (*parent_pending_threads)--;
    set_thread_state(WAIT);

    if (*parent_pending_threads == 0) {
        rc = pthread_cond_signal(parent_cond); // Wake up parent thread
        if (rc) perror("pthread_cond_signal");
    }
    pthread_mutex_unlock(&mutex);
}

void medoid_thread::run() {
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

void medoid_thread::wait() {
    int rc;
    rc = pthread_mutex_lock(&mutex);
    if (rc) perror("pthread_mutex_lock");

    while (state == WAIT) {
        //printf("Thread %d begin cond_wait\n", thd_id);
        rc = pthread_cond_wait(&cond, &mutex);
        if (rc) perror("pthread_cond_wait");
    }

    pthread_mutex_unlock(&mutex);
}

void medoid_thread::wake(thread_state_t state) {
    int rc;
    rc = pthread_mutex_lock(&mutex);
    if (rc) perror("pthread_mutex_lock");
    set_thread_state(state);
    if (state == thread_state_t::KMSPP_INIT)
        cuml_dist = 0;
    rc = pthread_mutex_unlock(&mutex);
    if (rc) perror("pthread_mutex_unlock");

    rc = pthread_cond_signal(&cond);
}

void* medoid_callback(void* arg) {
    medoid_thread* t = static_cast<medoid_thread*>(arg);
#ifdef USE_NUMA
    t->bind2node_id();
#endif

    while (true) { // So we can receive task after task
        if (t->get_state() == WAIT)
            t->wait();

        if (t->get_state() == EXIT) {// No more work to do
            //printf("Thread %d exiting ...\n", t->thd_id);
            break;
        }

        //printf("Thread %d awake and doing a run()\n", t->thd_id);
        t->run(); // else
    }

    // We've stopped running so exit
    pthread_exit(NULL);

#ifdef _WIN32
    return NULL;
#endif
}

void medoid_thread::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, medoid_callback, this);
    if (rc)
        throw kbase::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

const unsigned medoid_thread::
get_global_data_id(const unsigned row_id) const {
    return start_rid+row_id;
}

void medoid_thread::EM_step() {
    meta.num_changed = 0; // Always reset at the beginning of an EM-step
    local_clusters->clear();
    local_medoid_energy.assign(g_clusters->get_nclust(), 0);
    std::vector<double> medoid_ids;

    // NOTE: This has been bastardized
    for (auto id : g_clusters->get_num_members_v())
        medoid_ids.push_back(id);

    for (unsigned row = 0; row < nprocrows; row++) {
        // Choose row as new cluster center
        unsigned asgnd_clust = kbase::INVALID_CLUSTER_ID;
        double best, dist;
        dist = best = std::numeric_limits<double>::max();
        unsigned true_rid = get_global_data_id(row);

        for (unsigned clust_idx = 0;
                clust_idx < g_clusters->get_nclust(); clust_idx++) {

            dist = pw_dm->pw_get(true_rid, medoid_ids[clust_idx]);

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

void medoid_thread::medoid_step() {
    // Reset
    candidate_medoids.assign(g_clusters->get_nclust(), -1);
    candidate_medoid_energy.assign(g_clusters->get_nclust(),
            std::numeric_limits<double>::max());

    // NOTE: This has been bastardized
    std::vector<double> medoid_ids;
    for (auto id : g_clusters->get_num_members_v())
        medoid_ids.push_back(id);

    for (unsigned row = 0; row < nprocrows; row++) {
        unsigned true_rid = get_global_data_id(row);
        // What cluster the row is in
        unsigned cid = cluster_assignments[true_rid];
        unsigned medoid_id = medoid_ids[cid];
        double cluster_energy = global_medoid_energy[cid]; // Current energy
        double energy = 0;

        for (size_t sid = 0; sid < pw_dm->get_num_rows()+1; sid++) {
            if (cluster_assignments[sid] == cid && sid != true_rid) {
                // Check if it would reduce energy -- expensive
                energy += pw_dm->pw_get(true_rid, sid);
            }
        }

        // We have a possible better choice of medoid
        if (energy < candidate_medoid_energy[cid]) {
            candidate_medoid_energy[cid] = energy;
            candidate_medoids[cid] = true_rid;
        }
    }
}

const void medoid_thread::print_local_data() const {
    kbase::print_mat(local_data, nprocrows, ncol);
}
} // End namespace knor

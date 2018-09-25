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

namespace knor {
medoid_thread::medoid_thread(const int node_id, const unsigned thd_id,
        const unsigned start_rid,
        const unsigned nprocrows, const unsigned ncol,
        kbase::clusters::ptr g_clusters, unsigned* cluster_assignments,
        const std::string fn) : thread(node_id, thd_id, ncol,
            g_clusters->get_nclust(), cluster_assignments, start_rid, fn) {

            this->nprocrows = nprocrows;
            this->g_clusters = g_clusters;
            local_clusters =
                kbase::clusters::create(g_clusters->get_nclust(), ncol);

            set_data_size(sizeof(double)*nprocrows*ncol);
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
        case KMSPP_INIT:
            assert(0); // TODO: We don't need this case
            break;
        case EM:
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

void* callback(void* arg) {
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
    int rc = pthread_create(&hw_thd, NULL, callback, this);
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
    std::vector<double> local_dist;

    for (unsigned row = 0; row < nprocrows; row++) {

        // Choose row as new cluster center

        unsigned asgnd_clust = kbase::INVALID_CLUSTER_ID;
        double best, dist;
        dist = best = std::numeric_limits<double>::max();

        for (unsigned clust_idx = 0;
                clust_idx < g_clusters->get_nclust(); clust_idx++) {
            dist = kbase::dist_comp_raw<double>(&local_data[row*ncol],
                    &(g_clusters->get_means()[clust_idx*ncol]), ncol,
                    kbase::dist_t::EUCL);

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

const void medoid_thread::print_local_data() const {
    kbase::print_mat(local_data, nprocrows, ncol);
}
} // End namespace knor
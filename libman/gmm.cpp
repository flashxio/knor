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

#include "gmm.hpp"
#include "types.hpp"
#include "util.hpp"
#include "io.hpp"

namespace knor {
gmm::gmm(const int node_id, const unsigned thd_id,
        const unsigned start_rid,
        const unsigned nprocrows, const unsigned ncol,
        const std::string fn, kbase::dist_t dist_metric) :
            thread(node_id, thd_id, ncol,
            NULL, start_rid, fn, dist_metric) {

            this->nprocrows = nprocrows;
            local_clusters = nullptr;

            set_data_size(sizeof(double)*nprocrows*ncol);
#if VERBOSE
#ifndef
            std::cout << "Initializing gmm. Metadata: thd_id: "
                << this->thd_id << ", start_rid: " << this->start_rid <<
                ", node_id: " << this->node_id << ", nprocrows: " <<
                this->nprocrows << ", ncol: " << this->ncol << std::endl;
#endif
#endif
        }

void gmm::set_alg_metadata(unsigned k, base::dense_matrix<double>* mu_k,
        base::dense_matrix<double>** sigma_k, base::dense_matrix<double>* P_nk,
        double* P_k) {
    this->k = k;
    this->mu_k = mu_k;
    this->sigma_k = sigma_k;
    this->P_nk = P_nk;
}

void gmm::EM_step() {
}

void gmm::run() {
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

void gmm::sleep() {
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

void gmm::wait() {
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

void gmm::wake(thread_state_t state) {
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
    gmm* t = static_cast<gmm*>(arg);
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

void gmm::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback, this);
    if (rc)
        throw kbase::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

const unsigned gmm::
get_global_data_id(const unsigned row_id) const {
    return start_rid+row_id;
}

const void gmm::print_local_data() const {
    kbase::print_mat(local_data, nprocrows, ncol);
}
} // End namespace knor

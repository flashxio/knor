/*
 * Copyright 2016 neurodata (http://neurodata.io/)
 * Written by Disa Mhembere (disa@jhu.edu)
 *
 * This file is part of k-par-means
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

#include <cassert>

#include "base_kmeans_coordinator.hpp"
#include "base_kmeans_thread.hpp"
#include "util.hpp"

namespace kpmbase = kpmeans::base;

namespace kpmeans {
base_kmeans_coordinator::base_kmeans_coordinator(const std::string fn,
        const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const kpmbase::init_type_t it,
        const double tolerance, const kpmbase::dist_type_t dt) {

    this->fn = fn;
    this->nrow = nrow;
    this->ncol = ncol;
    this->k = k;
    kpmbase::assert_msg(k >= 1, "[FATAL]: 'k' must be >= 1");
    this->max_iters = max_iters;
    this->nnodes = nnodes;
    this->nthreads = static_cast<unsigned>(
            std::min(static_cast<size_t>(nthreads), this->nrow));

    this->_init_t = it;
    this->tolerance = tolerance;
    this->_dist_t = dt;
    num_changed = 0;
    pending_threads = 0;

    assert(cluster_assignments = new unsigned [nrow]);
    assert(cluster_assignment_counts = new size_t [k]);

    clear_cluster_assignments();
    std::fill(cluster_assignment_counts,
            cluster_assignment_counts+k, 0);

    // Threading
    pending_threads = 0; // NOTE: This must be initialized
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_init(&mutex, &mutex_attr);
    pthread_cond_init(&cond, NULL);
}

void base_kmeans_coordinator::wait4complete() {
    pthread_mutex_lock(&mutex);
    while (pending_threads != 0) {
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_mutex_unlock(&mutex);
}

void base_kmeans_coordinator::set_thread_data_ptr(double* allocd_data) {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it)
        (*it)->set_local_data_ptr(allocd_data);
}
} // End namespace kpmeans

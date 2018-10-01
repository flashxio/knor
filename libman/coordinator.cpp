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

#include <cassert>

#include "coordinator.hpp"
#include "thread.hpp"
#include "util.hpp"

namespace kbase = knor::base;

namespace knor {
coordinator::coordinator(const std::string fn,
        const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const kbase::init_t it,
        const double tolerance, const kbase::dist_t dt) {

    this->fn = fn;
    this->nrow = nrow;
    this->ncol = ncol;
    this->k = k;
    kbase::assert_msg(k >= 1, "[FATAL]: 'k' must be >= 1");
    this->max_iters = max_iters;
    this->nnodes = nnodes;
    this->nthreads = static_cast<unsigned>(
            std::min(static_cast<size_t>(nthreads), this->nrow));

    this->_init_t = it;
    this->tolerance = tolerance;
    this->_dist_t = dt;
    num_changed = 0;
    pending_threads = 0;

    cluster_assignments.resize(nrow);
    cluster_assignment_counts.resize(k);

    clear_cluster_assignments();
    std::fill(&cluster_assignment_counts[0],
            &cluster_assignment_counts[k], 0);

    // Threading
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_init(&mutex, &mutex_attr);
    pthread_cond_init(&cond, NULL);
}

void coordinator::wait4complete() {
    pthread_mutex_lock(&mutex);
    while (pending_threads != 0) {
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_mutex_unlock(&mutex);
}

void coordinator::set_thread_data_ptr(double* allocd_data) {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it)
        (*it)->set_local_data_ptr(allocd_data);
}
} // End namespace knor

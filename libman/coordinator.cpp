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
#include <algorithm>

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
        const double tolerance, const kbase::dist_t dt) : fn(fn), nrow(nrow),
    ncol(ncol), k(k), max_iters(max_iters), nnodes(nnodes),
    nthreads(static_cast<unsigned>(std::min(
                    static_cast<size_t>(nthreads), this->nrow))),
    _init_t(it), tolerance(tolerance), _dist_t(dt), num_changed(0),
    pending_threads(0) {

    kbase::assert_msg(k >= 1, "[FATAL]: 'k' must be >= 1");
    cluster_assignments.resize(nrow);
    clear_cluster_assignments();

    cluster_assignment_counts.resize(k);
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

std::pair<unsigned, unsigned>
coordinator::get_rid_len_tup(const unsigned thd_id) {
    unsigned rows_per_thread = nrow / nthreads;
    unsigned start_rid = (thd_id*rows_per_thread);

    if (thd_id == nthreads - 1)
        rows_per_thread += nrow % nthreads;
    return std::pair<unsigned, unsigned>(start_rid, rows_per_thread);
}


void coordinator::wake4run(const thread_state_t state) {
    pending_threads = nthreads;
    for (unsigned thd_id = 0; thd_id < threads.size(); thd_id++)
        threads[thd_id]->wake(state);
}

void coordinator::destroy_threads() {
    wake4run(EXIT);
}

// <Thread, within-thread-row-id>
const double* coordinator::get_thd_data(const unsigned row_id) const {
    // TODO: Cheapen
    unsigned parent_thd = std::upper_bound(thd_max_row_idx.begin(),
            thd_max_row_idx.end(), row_id) - thd_max_row_idx.begin();
    unsigned rows_per_thread = nrow/nthreads; // All but the last thread

    return &((threads[parent_thd]->get_local_data())
            [(row_id-(parent_thd*rows_per_thread))*ncol]);
}

void coordinator::set_thread_clust_idx(const unsigned clust_idx) {
    for (thread_iter it = threads.begin(); it != threads.end(); ++it)
        (*it)->set_clust_idx(clust_idx);
}

void const coordinator::print_thread_data() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it) {
#ifndef BIND
        printf("\nThd: %u\n", (*it)->get_thd_id());
#endif
        (*it)->print_local_data();
    }
}

// Testing
void const coordinator::print_thread_start_rids() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it) {
#ifndef BIND
        printf("\nThd: %u, start_rid: %lu\n", (*it)->get_thd_id(),
            (*it)->get_start_rid());
#endif
    }
}

void coordinator::set_thd_dist_v_ptr(double* v) {
    for (unsigned thd_id = 0; thd_id < threads.size(); thd_id++) {
        pthread_mutex_lock(&threads[thd_id]->get_lock());
        threads[thd_id]->set_dist_v_ptr(v);
        pthread_mutex_unlock(&threads[thd_id]->get_lock());
    }
}

double coordinator::reduction_on_cuml_sum() {
    double tot = 0;
    for (thread_iter it = threads.begin(); it != threads.end(); ++it)
        tot += (*it)->get_cuml_dist();
    return tot;
}

void coordinator::run_init() {
    switch(_init_t) {
        case kbase::init_t::RANDOM:
            random_partition_init();
            break;
        case kbase::init_t::FORGY:
            forgy_init();
            break;
        case kbase::init_t::PLUSPLUS:
            kmeanspp_init();
            break;
        case kbase::init_t::NONE:
            break;
        default:
            throw std::runtime_error("Unknown initialization type");
    }
}

coordinator::~coordinator() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it)
        (*it)->destroy_numa_mem();

    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&mutex);
    pthread_mutexattr_destroy(&mutex_attr);
    destroy_threads();
}

} // End namespace knor

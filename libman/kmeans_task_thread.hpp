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

#ifndef __KPM_KMEANS_TASK_THREAD_HPP__
#define __KPM_KMEANS_TASK_THREAD_HPP__

#include <atomic>

#include "base_kmeans_thread.hpp"

namespace kpmeans {
class task_queue;
class task;
    namespace base {
    class thd_safe_bool_vector;
    class prune_clusters;
    }
    namespace prune {
    class dist_matrix;
    }
}

namespace kpmbase = kpmeans::base;

namespace kpmeans { namespace prune {

class kmeans_task_thread : public kpmeans::base_kmeans_thread {
protected: // Lazy
    std::shared_ptr<kpmbase::prune_clusters> g_clusters; // Ptr to global cluster data
    unsigned start_rid; // The row id of the first item in this partition

    void* driver; // Hacky, but no time ...
    kpmeans::task_queue* tasks;
    kpmeans::task* curr_task;

    bool prune_init;
    std::shared_ptr<dist_matrix> dm; // global
    std::shared_ptr<kpmbase::thd_safe_bool_vector> recalculated_v; // global
    bool _is_numa;

    kmeans_task_thread(const int node_id, const unsigned thd_id,
            const unsigned start_rid, const unsigned nlocal_rows,
            const unsigned ncol,
            std::shared_ptr<kpmbase::prune_clusters> g_clusters,
            unsigned* cluster_assignments,
            const std::string fn);
public:
    static base_kmeans_thread::ptr create(const int node_id,
            const unsigned thd_id,
            const unsigned start_rid, const unsigned nlocal_rows,
            const unsigned ncol,
            std::shared_ptr<kpmbase::prune_clusters> g_clusters,
            unsigned* cluster_assignments, const std::string fn) {
        return base_kmeans_thread::ptr(
                new kmeans_task_thread(node_id, thd_id, start_rid,
                    nlocal_rows, ncol, g_clusters,
                    cluster_assignments, fn));
    }

    void start(const kpmeans::thread_state_t state);
    // Allocate and move data using this thread
    void EM_step();
    void kmspp_dist();
    const unsigned get_global_data_id(const unsigned row_id) const;
    void run();
    void wait();
    void wake(kpmeans::thread_state_t state);
    void request_task();
    void lock_sleep();
    void sleep();
    virtual bool try_steal_task();

    const void print_local_data() const;
    ~kmeans_task_thread();

    // Override
    void set_driver(void* driver) {
        this->driver = driver;
    }

    double* get_dist_v_ptr() { return &dist_v[0]; }

    void set_parent_cond(pthread_cond_t* cond) {
        parent_cond = cond;
    }

    void set_parent_pending_threads(std::atomic<unsigned>* ppt) {
        parent_pending_threads = ppt;
    }

    void set_prune_init(const bool prune_init) {
        this->prune_init = prune_init;
    }

    const bool is_prune_init() {
        return prune_init;
    }

    void set_recalc_v_ptr(std::shared_ptr<kpmbase::thd_safe_bool_vector>
            recalculated_v) {
        this->recalculated_v = recalculated_v;
    }

    void set_dist_mat_ptr(std::shared_ptr<dist_matrix> dm) {
        this->dm = dm;
    }

    kpmeans::task_queue* get_task_queue() {
      return tasks;
    }

    const unsigned get_thd_id() {
      return thd_id;
    }
};
} } // End namespace kpmeans, prune
#endif

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

#ifndef __KNOR_KMEANS_TASK_THREAD_HPP__
#define __KNOR_KMEANS_TASK_THREAD_HPP__

#include <atomic>
#include <random>

#include "task_thread.hpp"

namespace kbase = knor::base;

namespace knor {

class task_queue;
class task;

    namespace base {
    class thd_safe_bool_vector;
    class prune_clusters;
    }

    namespace prune {

class dist_matrix;

class kmeans_task_thread : public task_thread {
    using task_thread::task_thread;

public:
    static task_thread::ptr create(const int node_id,
            const unsigned thd_id,
            const unsigned start_rid, const unsigned nlocal_rows,
            const unsigned ncol,
            std::shared_ptr<kbase::prune_clusters> g_clusters,
            unsigned* cluster_assignments, const std::string fn,
            kbase::dist_t dist_metric) {
        return task_thread::ptr(
                new kmeans_task_thread(node_id, thd_id, start_rid,
                    nlocal_rows, ncol, g_clusters,
                    cluster_assignments, fn, dist_metric));
    }

    // Mini-batch
    void set_mb_perctg(const double mb_perctg) { this->mb_perctg = mb_perctg; }
    void mb_finalize_centroids(const double* eta);
    const size_t sample_size() const { return mb_selected.size(); }
    // End Mini-batch

    void start(const knor::thread_state_t state) override;
    // Allocate and move data using this thread
    void EM_step();
    void mb_EM_step();
    void kmspp_dist();
    void run() override;
    void wake(knor::thread_state_t state) override;
    virtual bool try_steal_task() override;
};
} } // End namespace knor, prune
#endif

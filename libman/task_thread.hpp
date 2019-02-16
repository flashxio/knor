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

#ifndef __KNOR_TASK_THREAD_HPP__
#define __KNOR_TASK_THREAD_HPP__

#include <atomic>
#include <random>

#include "thread.hpp"

namespace knor {
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

namespace kbase = knor::base;

namespace knor { namespace prune {

class task_thread : public knor::thread {
protected: // Lazy
    std::shared_ptr<kbase::prune_clusters> g_clusters; // Ptr to global cluster data

    void* driver; // Hacky, but no time ...
    knor::task_queue* tasks;
    knor::task* curr_task;

    bool prune_init;
    std::shared_ptr<dist_matrix> dm; // global
    std::shared_ptr<kbase::thd_safe_bool_vector> recalculated_v; // global
    bool _is_numa;

    // Mini-batch
    std::default_random_engine generator;
    std::uniform_real_distribution<double> ur_distribution;
    std::vector<unsigned> mb_selected; // Local ID of selected rows for mb
    double mb_perctg;

    task_thread(const int node_id, const unsigned thd_id,
            const unsigned start_rid, const unsigned nlocal_rows,
            const unsigned ncol,
            std::shared_ptr<kbase::prune_clusters> g_clusters,
            unsigned* cluster_assignments,
            const std::string fn, kbase::dist_t dist_metric);
public:
    typedef std::shared_ptr<task_thread> ptr;

    // Mini-batch
    void set_mb_perctg(const double mb_perctg) { this->mb_perctg = mb_perctg; }
    void mb_finalize_centroids(const double* eta);
    // End Mini-batch

    virtual void start(const knor::thread_state_t state) override
        { throw kbase::abstract_exception(); }
    // Allocate and move data using this thread
    virtual const unsigned get_global_data_id(const unsigned row_id)
        const override;
    virtual void run() override { throw kbase::abstract_exception(); }
    virtual void wait() override;
    virtual void wake(knor::thread_state_t state) override = 0;
    virtual void sleep() override;

    virtual void request_task();
    virtual void lock_sleep();
    virtual bool try_steal_task() override = 0;

    ~task_thread();

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

    void set_prune_init(const bool prune_init) override {
        this->prune_init = prune_init;
    }

    const bool is_prune_init() {
        return prune_init;
    }

    void set_recalc_v_ptr(std::shared_ptr<kbase::thd_safe_bool_vector>
            recalculated_v) override {
        this->recalculated_v = recalculated_v;
    }

    void set_dist_mat_ptr(std::shared_ptr<dist_matrix> dm) override {
        this->dm = dm;
    }

    knor::task_queue* get_task_queue() override {
      return tasks;
    }

    const unsigned get_thd_id() {
      return thd_id;
    }
};
} } // End namespace knor, prune
#endif

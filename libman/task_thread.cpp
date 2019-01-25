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

#include "task_thread.hpp"
#include "task_queue.hpp"
#include "clusters.hpp"
#include "thd_safe_bool_vector.hpp"
#include "dist_matrix.hpp"
#include "util.hpp"

#if 1
#endif
namespace knor { namespace prune {

task_thread::task_thread(const int node_id, const unsigned thd_id,
        const unsigned start_rid, const unsigned nlocal_rows,
        const unsigned ncol,
        std::shared_ptr<kbase::prune_clusters> g_clusters,
        unsigned* cluster_assignments,
        const std::string fn, kbase::dist_t dist_metric):
            thread(node_id, thd_id, ncol,
            cluster_assignments, start_rid, fn, dist_metric),
        g_clusters(g_clusters), prune_init(true), _is_numa(false) {

                ur_distribution =
                    std::uniform_real_distribution<double>(0.0, 1.0);
                // Init task queue
                tasks = new task_queue();

                tasks->set_start_rid(start_rid);
                tasks->set_nrow(nlocal_rows);
                tasks->set_ncol(ncol);
                local_clusters =
                    kbase::clusters::create(g_clusters->get_nclust(), ncol);

                set_data_size(sizeof(double)*nlocal_rows*ncol);
#if VERBOSE
#ifndef
                std::cout << "Init task_thread. Metadata: thd_id: "
                    << this->thd_id << ", start_rid: " << this->start_rid <<
                    ", node_id: " << this->node_id << ", nlocal_rows: " <<
                    nlocal_rows << ", ncol: " << this->ncol << std::endl;
#endif
#endif
            }

void task_thread::request_task() {
    int rc;
    rc = pthread_mutex_lock(&mutex);
    if (rc) perror("pthread_mutex_lock");

    if (tasks->has_task()) {
        // Grab another task but drop the old task
        if (curr_task)
            delete curr_task;

        /*printf("Thd :%u getting own task with rid: %u\n",
            get_thd_id(), tasks->get_nxt_rid());*/

        curr_task = tasks->get_task();
        assert(curr_task->get_nrow() <= tasks->get_nrow());

        // FIXME: someone got the last task
        //printf("request_task: Thd: %u, Task ==> ", get_thd_id()); curr_task.print();
        kbase::assert_msg(curr_task->get_nrow(), "FIXME: Empty task");
        pthread_mutex_unlock(&mutex);
    }
#if 0
    else {
        if (try_steal_task())
            // Try to steal
        }
#else
    else {
        sleep();
        pthread_mutex_unlock(&mutex);
    }
#endif
}

void task_thread::lock_sleep() {
    int rc;
    rc = pthread_mutex_lock(&mutex);
    if (rc) perror("pthread_mutex_lock");

    (*parent_pending_threads)--;
    set_thread_state(WAIT);

    if (*parent_pending_threads == 0) {
        rc = pthread_cond_signal(parent_cond); // Wake up parent thread
        if (rc) perror("pthread_cond_signal");
    }
    rc = pthread_mutex_unlock(&mutex);
    if (rc) perror("pthread_mutex_unlock");
}

// Assumes caller has lock already ... or else ...
void task_thread::sleep() {
    (*parent_pending_threads)--;
    set_thread_state(WAIT);

    if (*parent_pending_threads == 0) {
        int rc = pthread_cond_signal(parent_cond); // Wake up parent thread
        if (rc) perror("pthread_cond_signal");
    }
}

void task_thread::wait() {
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

const unsigned task_thread::get_global_data_id(const unsigned row_id) const {
    return row_id + curr_task->get_start_rid();
}

task_thread::~task_thread() {
  delete tasks;
}
} } // End namespace knor, prune

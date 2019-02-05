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

#include "kmeans_task_thread.hpp"
#include "task_queue.hpp"
#include "kmeans_task_coordinator.hpp"
#include "clusters.hpp"
#include "thd_safe_bool_vector.hpp"
#include "dist_matrix.hpp"

namespace knor { namespace prune {

/* \brief NUMA aware or oblivious task stealing
   \param return true if I got a task tasks
 **/
bool kmeans_task_thread::try_steal_task() {
  std::vector<std::shared_ptr<knor::thread> > workers =
    (static_cast<kmeans_task_coordinator*>(driver))->get_threads(); // Me included

  bool one_locked;
  do {
      one_locked = false;
      for (unsigned i = 0; i < workers.size(); i++) {
          if (i != get_thd_id()) { // Can't steal from myself & I'm already done

              int rc = pthread_mutex_trylock(&workers[i]->get_lock());

              if (EXIT_SUCCESS == rc) { // Acquired the lock
                  if (workers[i]->get_task_queue()->has_task()) {
                      // TODO: Actually steal task
                      //printf("Thread %u stealing task from thread %u!\n", thd_id, i);

                      pthread_mutex_unlock(&workers[i]->get_lock());
                      //printf("T: %u stole & released T: %u's lock\n", thd_id, i);
                      return false; // TODO: change to true
                  } else { // Thread has no tasks to give
                      pthread_mutex_unlock(&workers[i]->get_lock());
                      //printf("T: %u couldn't steal & released T: %u's lock\n", thd_id, i);
                      continue;
                  }
              }

              // Didn't get the lock
              if (rc == EBUSY) { // Move on if you can't get the lock
                  //printf("T: %u says T:%u is busy \n", thd_id, i);
                  if (workers[i]->get_task_queue()->has_task())
                      one_locked = true;

                  continue;
              }
          }
      }
  } while (one_locked);
  return false;
}

void kmeans_task_thread::run() {
    switch(state) {
        case TEST:
            test();
            lock_sleep();
            break;
        case ALLOC_DATA:
            numa_alloc_mem();
            tasks->set_data_ptr(local_data); // We now have real data
            lock_sleep();
            break;
        case KMSPP_INIT:
            kmspp_dist();
            request_task();
            break;
        case EM: /* Super-E-step */
            EM_step();
            request_task();
            break;
        case MB_EM: /* Mini-batch Super-E-step */
            mb_EM_step();
            request_task();
            break;
        case EXIT:
            throw kbase::thread_exception(
                    "Thread state is EXIT but running!\n");
        default:
            throw kbase::thread_exception("Unknown thread state\n");
    }
}

void kmeans_task_thread::wake(thread_state_t state) {
    int rc;
    rc = pthread_mutex_lock(&mutex);
    if (rc) perror("pthread_mutex_lock");
    set_thread_state(state);

    if (state == thread_state_t::EM ||
            state == thread_state_t::KMSPP_INIT ||
            state == thread_state_t::MB_EM) {
        // Threads only sleep if they AND all other threads have no tasks
        tasks->reset(); // NOTE: Only place this is reset
        curr_task = tasks->get_task();
        assert(curr_task->get_nrow() <= tasks->get_nrow());

        // NOTE: These are exceptions to the rule & therefore not good
        if (state == thread_state_t::EM || state == thread_state_t::MB_EM) {
            meta.num_changed = 0; // Always reset at the beginning of an EM-step
        }

        if (state == thread_state_t::KMSPP_INIT)
            cuml_dist = 0;

        local_clusters->clear();

        //printf("wake: Thd: %u, Task ==> ", get_thd_id()); curr_task.print();
    }

    rc = pthread_mutex_unlock(&mutex);
    if (rc) perror("pthread_mutex_unlock");

    rc = pthread_cond_signal(&cond);
}

void kmeans_task_thread::start(const thread_state_t state=WAIT) {
    //printf("Thread %d started ...\n", thd_id);
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback<kmeans_task_thread>, this);
    if (rc)
        throw kbase::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

void kmeans_task_thread::mb_finalize_centroids(const double* eta) {
    // At least it is sequential access
    for (unsigned local_rid : mb_selected) {
        auto g_rid = start_rid + local_rid;

        auto cid = cluster_assignments[g_rid];
        assert(cid < g_clusters->get_nclust());
        g_clusters->scale_centroid(eta[cid], cid, &(local_data[local_rid*ncol]));
    }

    // Clear mb_selected for the next iteration
    mb_selected.clear();
}

void kmeans_task_thread::mb_EM_step() {
    for (unsigned row = 0; row < curr_task->get_nrow(); row++) {
        if (ur_distribution(generator) > mb_perctg)
            continue; // Sample rows

        unsigned true_row_id = get_global_data_id(row);

        mb_selected.push_back(true_row_id - start_rid); // Local rid

        for (unsigned clust_idx = 0;
                clust_idx < g_clusters->get_nclust(); clust_idx++) {
            double dist = kbase::dist_comp_raw<double>(
                    &curr_task->get_data_ptr()[row*ncol],
                    &(g_clusters->get_means()[clust_idx*ncol]), ncol,
                    dist_metric);

            if (dist < dist_v[true_row_id]) {
                dist_v[true_row_id] = dist;
                cluster_assignments[true_row_id] = clust_idx;
            }
        }
    }
}

void kmeans_task_thread::EM_step() {
    for (unsigned row = 0; row < curr_task->get_nrow(); row++) {
        unsigned true_row_id = get_global_data_id(row);
        unsigned old_clust = cluster_assignments[true_row_id];

        if (prune_init) {
            double dist = std::numeric_limits<double>::max();

            for (unsigned clust_idx = 0;
                    clust_idx < g_clusters->get_nclust(); clust_idx++) {
                dist = kbase::dist_comp_raw<double>(
                        &curr_task->get_data_ptr()[row*ncol],
                        &(g_clusters->get_means()[clust_idx*ncol]), ncol,
                        dist_metric);

                if (dist < dist_v[true_row_id]) {
                    dist_v[true_row_id] = dist;
                    cluster_assignments[true_row_id] = clust_idx;
                }
            }

        } else {
            recalculated_v->set(true_row_id, false);
            dist_v[true_row_id] +=
                g_clusters->get_prev_dist(cluster_assignments[true_row_id]);

            if (dist_v[true_row_id] <=
                    g_clusters->get_s_val(cluster_assignments[true_row_id])) {
                // Skip all rows
            } else {
                for (unsigned clust_idx = 0;
                        clust_idx < g_clusters->get_nclust(); clust_idx++) {

                    if (dist_v[true_row_id] <= dm->get(cluster_assignments
                                [true_row_id], clust_idx)) {
                        // Skip this cluster
                        continue;
                    }

                    if (!recalculated_v->get(true_row_id)) {
                        dist_v[true_row_id] = kbase::dist_comp_raw<double>(
                                &curr_task->get_data_ptr()[row*ncol],
                                &(g_clusters->get_means()[cluster_assignments
                                    [true_row_id]*ncol]), ncol,
                                dist_metric);
                        recalculated_v->set(true_row_id, true);
                    }

                    if (dist_v[true_row_id] <=
                            dm->get(cluster_assignments[true_row_id], clust_idx)) {
                        // Skip this cluster
                        continue;
                    }

                    // Track 5
                    double jdist = kbase::dist_comp_raw(
                            &curr_task->get_data_ptr()[row*ncol],
                            &(g_clusters->get_means()[clust_idx*ncol]), ncol,
                            dist_metric);

                    if (jdist < dist_v[true_row_id]) {
                        dist_v[true_row_id] = jdist;
                        cluster_assignments[true_row_id] = clust_idx;
                    }
                } // endfor
            }
        }

        assert(cluster_assignments[true_row_id] >= 0 &&
                cluster_assignments[true_row_id] < g_clusters->get_nclust());

        if (prune_init) {
            meta.num_changed++;
            local_clusters->add_member(&(curr_task->get_data_ptr()[row*ncol]),
                    cluster_assignments[true_row_id]);
        } else if (old_clust != cluster_assignments[true_row_id]) {
            meta.num_changed++;
            local_clusters->swap_membership(
                    &(curr_task->get_data_ptr()[row*ncol]),
                    old_clust, cluster_assignments[true_row_id]);
        }
    }
}

/** Method for a distance computation vs a single cluster.
 * Used in kmeans++ init
 */
void kmeans_task_thread::kmspp_dist() {
    unsigned clust_idx = meta.clust_idx;
    for (unsigned row = 0; row < curr_task->get_nrow(); row++) {
        unsigned true_row_id = get_global_data_id(row);

        double dist = kbase::dist_comp_raw<double>(
                &(curr_task->get_data_ptr()[row*ncol]),
                &((g_clusters->get_means())[clust_idx*ncol]), ncol,
                dist_metric);

        if (dist < dist_v[true_row_id]) { // Found a closer cluster than before
            dist_v[true_row_id] = dist;
            cluster_assignments[true_row_id] = clust_idx;
        }

        cuml_dist += dist_v[true_row_id];
    }
}
} } // End namespace knor, prune

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

#include "kmeans_task_thread.hpp"
#include "task_queue.hpp"
#include "kmeans_task_coordinator.hpp"
#include "kcommon.hpp"

namespace kpmeans { namespace prune {

kmeans_task_thread::kmeans_task_thread(const int node_id, const unsigned thd_id,
        const unsigned start_rid, const unsigned nlocal_rows,
        const unsigned ncol,
        std::shared_ptr<kpmbase::prune_clusters> g_clusters,
        unsigned* cluster_assignments,
        const std::string fn) : base_kmeans_thread(node_id, thd_id, ncol,
            g_clusters->get_nclust(), cluster_assignments, start_rid, fn) {

            this->g_clusters = g_clusters;
            // Init task queue
            tasks = new task_queue();

            tasks->set_start_rid(start_rid);
            tasks->set_nrow(nlocal_rows);
            tasks->set_ncol(ncol);
            prune_init = true;
            _is_numa = false; // TODO: param this
            local_clusters =
                kpmbase::clusters::create(g_clusters->get_nclust(), ncol);

            set_data_size(sizeof(double)*nlocal_rows*ncol);
#if VERBOSE
            BOOST_LOG_TRIVIAL(info) << "Initializing thread. Metadata: thd_id: "
                << this->thd_id << ", start_rid: " << this->start_rid <<
                ", node_id: " << this->node_id << ", nlocal_rows: " <<
                nlocal_rows << ", ncol: " << this->ncol;
#endif
        }

void kmeans_task_thread::request_task() {
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
        BOOST_VERIFY(curr_task->get_nrow() <= tasks->get_nrow());

        // FIXME: someone got the last task
        //printf("request_task: Thd: %u, Task ==> ", get_thd_id()); curr_task.print();
        BOOST_ASSERT_MSG(curr_task->get_nrow(), "FIXME: Empty task");
        pthread_mutex_unlock(&mutex);
    }
#if 0
    else {
        pthread_mutex_unlock(&mutex);
        if (try_steal_task()) {

        } else {
          rc = pthread_mutex_lock(&mutex);
          if (rc) perror("pthread_mutex_lock");
          sleep();
          pthread_mutex_unlock(&mutex);
        }
    }
#else
    else {
        sleep();
        pthread_mutex_unlock(&mutex);
    }
#endif
}

/* \brief NUMA aware or oblivious task stealing
   \param return true if I got a task tasks
 **/
#if 0
bool kmeans_task_thread::try_steal_task() {
  std::vector<std::shared_ptr<prune::kmeans_task_thread> > workers =
    (static_cast<kmeans_task_coordinator*>(driver))->get_threads(); // Myself included
  for (unsigned i = 0; i < workers.size(); i++) {
    if (i != get_thd_id()) { // If I'm stealing I have no work
      // TODO: No lock yet ...
      if (workers[i]->get_task_queue()->has_task()) { // TODO: No preference yet
        int ret = pthread_mutex_trylock(&workers[i]->get_lock());

        if (ret != EBUSY) {
          printf("Thread %u stealing task from thread %u\n", get_thd_id(), i);

          if (NULL != curr_task)
            delete curr_task; // Free me breh

          curr_task = workers[i]->get_task_queue()->get_task(); // Stolen a task
          pthread_mutex_unlock(&workers[i]->get_lock());
          break;
        }
      }
    }
  }

  if (!curr_task->get_nrow()) {
    printf("Thread %u failed to steal task!\n", get_thd_id());
    return false; // Can happen if last task was stolen
  }
  return true;
}

#else

bool kmeans_task_thread::try_steal_task() {
  std::vector<std::shared_ptr<kpmeans::base_kmeans_thread> > workers =
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
#endif

void kmeans_task_thread::lock_sleep() {
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
void kmeans_task_thread::sleep() {
    (*parent_pending_threads)--;
    set_thread_state(WAIT);

    if (*parent_pending_threads == 0) {
        int rc = pthread_cond_signal(parent_cond); // Wake up parent thread
        if (rc) perror("pthread_cond_signal");
    }
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
        case EXIT:
            throw kpmbase::thread_exception(
                    "Thread state is EXIT but running!\n");
        default:
            throw kpmbase::thread_exception("Unknown thread state\n");
    }
}

void kmeans_task_thread::wait() {
    int rc;
    rc = pthread_mutex_lock(&mutex);
    if (rc) perror("pthread_mutex_lock");

    while (state == WAIT) {
        //printf("Thread %d begin cond_wait\n", thd_id);
        rc = pthread_cond_wait(&cond, &mutex);
        if (rc) perror("pthread_cond_wait");
    }

    rc = pthread_mutex_unlock(&mutex);
    if (rc) perror("pthread_mutex_unlock");
}

void kmeans_task_thread::wake(thread_state_t state) {
    int rc;
    rc = pthread_mutex_lock(&mutex);
    if (rc) perror("pthread_mutex_lock");
    set_thread_state(state);

    if (state == thread_state_t::EM ||
            state == thread_state_t::KMSPP_INIT) {
        // Threads only sleep if they AND all other threads have no tasks
        tasks->reset(); // NOTE: Only place this is reset
        curr_task = tasks->get_task();
        BOOST_VERIFY(curr_task->get_nrow() <= tasks->get_nrow());

        // TODO: These are exceptions to the rule & therefore not good
        if (state == thread_state_t::EM)
            meta.num_changed = 0; // Always reset at the beginning of an EM-step
        if (state == thread_state_t::KMSPP_INIT)
            cuml_dist = 0;

        local_clusters->clear();

        //printf("wake: Thd: %u, Task ==> ", get_thd_id()); curr_task.print();
    }

    rc = pthread_mutex_unlock(&mutex);
    if (rc) perror("pthread_mutex_unlock");

    rc = pthread_cond_signal(&cond);
}

void* callback(void* arg) {
    kmeans_task_thread* t = static_cast<kmeans_task_thread*>(arg);
#ifdef LINUX
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
}

void kmeans_task_thread::start(const thread_state_t state=WAIT) {
    //printf("Thread %d started ...\n", thd_id);
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback, this);
    if (rc)
        throw kpmbase::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

const unsigned kmeans_task_thread::
get_global_data_id(const unsigned row_id) const {
    return row_id + curr_task->get_start_rid();
}

void kmeans_task_thread::EM_step() {
    for (unsigned row = 0; row < curr_task->get_nrow(); row++) {
        unsigned true_row_id = get_global_data_id(row);
        unsigned old_clust = cluster_assignments[true_row_id];

        if (prune_init) {
            double dist = std::numeric_limits<double>::max();

            for (unsigned clust_idx = 0;
                    clust_idx < g_clusters->get_nclust(); clust_idx++) {
                dist = kpmbase::dist_comp_raw<double>(
                        &curr_task->get_data_ptr()[row*ncol],
                        &(g_clusters->get_means()[clust_idx*ncol]), ncol,
                        kpmbase::dist_type_t::EUCL);

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
                        dist_v[true_row_id] = kpmbase::dist_comp_raw<double>(
                                &curr_task->get_data_ptr()[row*ncol],
                                &(g_clusters->get_means()[cluster_assignments
                                    [true_row_id]*ncol]), ncol,
                                kpmbase::dist_type_t::EUCL);
                        recalculated_v->set(true_row_id, true);
                    }

                    if (dist_v[true_row_id] <=
                            dm->get(cluster_assignments[true_row_id], clust_idx)) {
                        // Skip this cluster
                        continue;
                    }

                    // Track 5
                    double jdist = kpmbase::dist_comp_raw(
                            &curr_task->get_data_ptr()[row*ncol],
                            &(g_clusters->get_means()[clust_idx*ncol]), ncol,
                            kpmbase::dist_type_t::EUCL);

                    if (jdist < dist_v[true_row_id]) {
                        dist_v[true_row_id] = jdist;
                        cluster_assignments[true_row_id] = clust_idx;
                    }
                } // endfor
            }
        }

        BOOST_VERIFY(cluster_assignments[true_row_id] >= 0 &&
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

        double dist = kpmbase::dist_comp_raw<double>(
                &(curr_task->get_data_ptr()[row*ncol]),
                &((g_clusters->get_means())[clust_idx*ncol]), ncol,
                kpmbase::dist_type_t::EUCL);

        if (dist < dist_v[true_row_id]) { // Found a closer cluster than before
            dist_v[true_row_id] = dist;
            cluster_assignments[true_row_id] = clust_idx;
        }

        cuml_dist += dist_v[true_row_id];
    }
}

const void kmeans_task_thread::print_local_data() const {
    kpmbase::print_mat(local_data,
            (get_data_size()/(sizeof(double)*ncol)), ncol);
}

kmeans_task_thread::~kmeans_task_thread() {
  delete tasks;
}
} } // End namespace kpmeans, prune

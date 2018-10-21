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

#include "skmeans_coordinator.hpp"
#include "skmeans.hpp"
#include "clusters.hpp"
#include "io.hpp"

namespace knor {

    skmeans_coordinator::skmeans_coordinator(
            const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const base::init_t it,
                const double tolerance, const base::dist_t dt) {
        /* :
           fn(fn), nrow(nrow), ncol(ncol), k(k), max_iters(max_iters),
           nnodes(nnodes), _init_t(it), tolerance(tolerance),
           _dist_t(dt), num_changed(0), pending_threads(0)*/
        /*

           kmeans_coordinator(fn, nrow, ncol, k, max_iters,
           nnodes, nthreads, centers, it, tolerance, dt) {
           }
           */

            kbase::assert_msg(k >= 1, "[FATAL]: 'k' must be >= 1");
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

    void skmeans_coordinator::build_thread_state() {
        // NUMA node affinity binding policy is round-robin
        printf("skmeans_coordinator::build_thread_state()\n");
        unsigned thds_row = nrow / nthreads;
        for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
            std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
            thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
            threads.push_back(skmeans::create((thd_id % nnodes),
                        thd_id, tup.first, tup.second,
                        ncol, cltrs, &cluster_assignments[0], fn));
            threads[thd_id]->set_parent_cond(&cond);
            threads[thd_id]->set_parent_pending_threads(&pending_threads);
            threads[thd_id]->start(WAIT); // Thread puts itself to sleep
        }
    }

    void skmeans_coordinator::bounds_reduction() {
        g_feature_max.assign(ncol, std::numeric_limits<double>::min());
        g_feature_min.assign(ncol, std::numeric_limits<double>::max());

        for (auto th : threads) {
            auto t = std::static_pointer_cast<skmeans>(th);

            auto min_fv = t->get_min_feature_val();
            auto max_fv = t->get_max_feature_val();
            for (size_t col = 0; col < ncol; col++) {
                if (min_fv[col] < g_feature_min[col])
                    g_feature_min[col] = min_fv[col];
                if (max_fv[col] > g_feature_max[col])
                    g_feature_max[col] = max_fv[col];
            }
        }

        // Pass the final min/max to each thread
        for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
            auto t = std::static_pointer_cast<skmeans>(threads[thd_id]);
            t->get_min_feature_val() = g_feature_min;
            t->get_max_feature_val() = g_feature_max;
        }
    }

    void skmeans_coordinator::preprocess_data() {
        printf("\n\nskmeans_coordinator::preprocess_data\n\n!");
        wake4run(BOUNDS);
        wait4complete();

        bounds_reduction();

        wake4run(NORMALIZE_DATA);
        wait4complete();
    }

kbase::cluster_t skmeans_coordinator::run(
        double* allocd_data, const bool numa_opt) {
#ifdef PROFILER
    ProfilerStart("libman/skmeans_coordinator.perf");
#endif

    if (!numa_opt && NULL == allocd_data) {
        wake4run(ALLOC_DATA);
        wait4complete();
    } else if (allocd_data) { // No NUMA opt
        set_thread_data_ptr(allocd_data);
    } // Do nothing for numa_opt .. done in binding/knori.hpp

    std::cout << "Data before:\n";
    print_thread_data();

    preprocess_data();
    std::cout << "\n\nData after:\n";
    print_thread_data();
    exit(911);
    ///////////////////////////////////////////////////////////////////////////

    struct timeval start, end;
    gettimeofday(&start , NULL);
    run_init(); // Initialize clusters

    // Run kmeans loop
    bool converged = false;
    size_t iter = 0;

    if (max_iters > 0)
        iter++;

    while (iter <= max_iters && max_iters > 0) {
        if (iter == 1)
            clear_cluster_assignments();

        wake4run(EM);
        wait4complete();

        update_clusters();

#if VERBOSE
#ifndef BIND
        printf("Cluster assignment counts: \n");
#endif
        kbase::print_vector(cluster_assignment_counts);
#endif

        if (num_changed == 0 ||
                ((num_changed/(double)nrow)) <= tolerance) {
            converged = true;
            break;
        }
        iter++;
    }
#ifdef PROFILER
    ProfilerStop();
#endif

    gettimeofday(&end, NULL);
#ifndef BIND
    printf("\n\nAlgorithmic time taken = %.6f sec\n",
        kbase::time_diff(start, end));
    printf("\n******************************************\n");
#endif
    if (converged) {
#ifndef BIND
        printf("K-means converged in %lu iterations\n", iter);
#endif
    } else {
#ifndef BIND
        printf("[Warning]: K-means failed to converge in %lu"
            " iterations\n", iter);
#endif
    }

#ifndef BIND
    printf("Final cluster counts: \n");
    kbase::print_vector(cluster_assignment_counts);
    printf("\n******************************************\n");
#endif

    return kbase::cluster_t(this->nrow, this->ncol, iter, this->k,
            &cluster_assignments[0], &cluster_assignment_counts[0],
            cltrs->get_means());
}
}
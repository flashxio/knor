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

#include <stdexcept>

#include "kmeans_task_coordinator.hpp"
#include "kmeans_task_thread.hpp"
#include "dist_matrix.hpp"
#include "clusters.hpp"
#include "thd_safe_bool_vector.hpp"
#include "linalg.hpp"

#include "task_queue.hpp"

namespace kbase = knor::base;

namespace knor { namespace prune {
kmeans_task_coordinator::kmeans_task_coordinator(const std::string fn,
        const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const kbase::init_t it,
        const double tolerance, const kbase::dist_t dt) :
    coordinator(fn, nrow, ncol, k, max_iters,
            nnodes, nthreads, centers, it, tolerance, dt) {

        cltrs = kbase::prune_clusters::create(k, ncol);

        inited = false;
        if (centers) {
            if (it == kbase::init_t::NONE) {
                cltrs->set_mean(centers);
            } else {
#ifndef BIND
                printf("[WARNING]: Both init centers"
                        "provided & non-NONE init method specified\n");
#endif
            }
        }

        // For pruning
        recalculated_v = kbase::thd_safe_bool_vector::create(nrow, false);
        dist_v.resize(nrow);
        std::fill(&dist_v[0], &dist_v[nrow], std::numeric_limits<double>::max());
        dm = prune::dist_matrix::create(k);
        build_thread_state();
}

void kmeans_task_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;
    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(prune::kmeans_task_thread::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, cltrs, &cluster_assignments[0], fn, _dist_t));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
        std::static_pointer_cast<task_thread>(threads[thd_id])
            ->set_driver(this); // For computation stealing
    }
}

void kmeans_task_coordinator::set_prune_init(const bool prune_init) {
    for (thread_iter it = threads.begin(); it != threads.end(); ++it)
        (*it)->set_prune_init(prune_init);
}

void kmeans_task_coordinator::set_global_ptrs() {
    for (thread_iter it = threads.begin(); it != threads.end(); ++it) {
        pthread_mutex_lock(&mutex);
        (*it)->set_dist_v_ptr(&dist_v[0]);
        (*it)->set_recalc_v_ptr(recalculated_v);
        (*it)->set_dist_mat_ptr(dm);
        pthread_mutex_unlock(&mutex);
    }
}

void kmeans_task_coordinator::mb_iteration_end() {
    // Fine at for O(n)
    std::vector<double>v; v.assign(k, 0);// Use std::fill
    for (size_t rid = 0; rid < nrow; rid++) {
        auto cid = cluster_assignments[rid];
        if (cid != base::INVALID_CLUSTER_ID) // Skip those not sampled
            v[cid]++;
    }

    // Fine at O(k)
    for (size_t cid = 0; cid < k; cid++)
        v[cid] = 1.0/v[cid];

    // BAD: Serial O(t*b)
    // NOTE: This updates global clusters
#if 0
    size_t total_procd = 0;
#endif
    for (auto const& th : threads) {
#if 0
        total_procd += std::static_pointer_cast<kmeans_task_thread>(th)
            ->sample_size();
#endif
        std::static_pointer_cast<kmeans_task_thread>(th)
            ->mb_finalize_centroids(&v[0]);
    }

#if 0
    printf("Total samples for iteration: %lu\n", total_procd);
#endif
    // Reset the distances
    std::fill(&dist_v[0], &dist_v[nrow], std::numeric_limits<double>::max());
}

void kmeans_task_coordinator::update_clusters(const bool prune_init) {
    if (prune_init) {
#ifndef BIND
        printf("Clearing because of init ..\n");
#endif
        cltrs->clear();
    } else {
        cltrs->set_prev_means();
        cltrs->unfinalize_all();
    }

    for (thread_iter it = threads.begin(); it != threads.end(); ++it) {
        // Updated the changed cluster count
        num_changed += (*it)->get_num_changed();
        cltrs->peq((*it)->get_local_clusters());
    }

    unsigned chk_nmemb = 0;
    for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) {
        cltrs->finalize(clust_idx);
        cltrs->set_prev_dist(
                kbase::eucl_dist(&(cltrs->get_means()[clust_idx*ncol]),
                &(cltrs->get_prev_means()[clust_idx*ncol]), ncol), clust_idx);
#if VERBOSE
#ifndef BIND
        printf("Dist to prev mean for c: %u is %.6f\n",
                clust_idx, cltrs->get_prev_dist(clust_idx));
#endif
#endif
        cluster_assignment_counts[clust_idx] = cltrs->get_num_members(clust_idx);
        chk_nmemb += cluster_assignment_counts[clust_idx];
    }
    assert(chk_nmemb == nrow);

#if KM_TEST
#ifndef BIND
    printf("Global number of changes: %lu\n", num_changed);
#endif
#endif
}

void kmeans_task_coordinator::set_thread_data_ptr(double* allocd_data) {

    coordinator::set_thread_data_ptr(allocd_data);
    // We must also set the pointer for the task queues
    set_task_data_ptrs();
}

void kmeans_task_coordinator::kmeanspp_init() {
    struct timeval start, end;
    gettimeofday(&start , NULL);

    // Choose c1 uniformly at random
    if (!inited)
        ui_distribution = std::uniform_int_distribution<unsigned>(0, nrow-1);

    unsigned selected_idx = ui_distribution(generator);

    cltrs->set_mean(get_thd_data(selected_idx), 0);
    dist_v[selected_idx] = 0.0;
    assert(cluster_assignments.size() == nrow);

    if (cluster_assignments.size() != nrow)
        cluster_assignments.assign(nrow, base::INVALID_CLUSTER_ID);

    cluster_assignments[selected_idx] = 0;

#if KM_TEST
#ifndef BIND
    printf("Choosing %u as center k = 0\n", selected_idx);
#endif
#endif
    unsigned clust_idx = 0; // The number of clusters assigned

    if (!inited) {
        ur_distribution = std::uniform_real_distribution<double>(0.0, 1.0);
        inited = true;
    }

    // Choose next center c_i with weighted prob
    while (true) {
        set_thread_clust_idx(clust_idx); // Set the current cluster index
        wake4run(KMSPP_INIT); // Run || distance comp to clust_idx
        wait4complete();
        double cuml_dist = reduction_on_cuml_sum(); // Sum the per thread cumulative dists

        cuml_dist = (cuml_dist * ur_distribution(generator)) / (RAND_MAX - 1.0);
        if (++clust_idx >= k)  // No more centers needed
            break;

        for (size_t row = 0; row < nrow; row++) {
            cuml_dist -= dist_v[row];
            if (cuml_dist <= 0) {
#if KM_TEST
#ifndef BIND
                printf("Choosing %lu as center k = %u\n",
                        row, clust_idx);
#endif
#endif
                cltrs->set_mean(get_thd_data(row), clust_idx);
                cluster_assignments[row] = clust_idx;
                break;
            }
        }
        assert(cuml_dist <= 0);
    }

#if VERBOSE
#ifndef BIND
    printf("\nCluster centers after kmeans++\n");
    cltrs->print_means();
#endif
#endif
    gettimeofday(&end, NULL);
#ifndef BIND
    printf("Initialization time: %.6f sec\n",
        kbase::time_diff(start, end));
#endif
}

void kmeans_task_coordinator::random_partition_init() {
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, k-1);

    for (unsigned row = 0; row < nrow; row++) {
        unsigned asgnd_clust = distribution(generator);
        const double* dp = get_thd_data(row);

        cltrs->add_member(dp, asgnd_clust);
        cluster_assignments[row] = asgnd_clust;
    }

    cltrs->finalize_all();

#if VERBOSE
#ifndef BIND
    printf("After rand paritions cluster_asgns: \n");
    print<unsigned>(cluster_assignments);
#endif
#endif
}

void kmeans_task_coordinator::forgy_init() {
#if 1
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, nrow-1);

    for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) { // 0...k
        unsigned rand_idx = distribution(generator);
        cltrs->set_mean(get_thd_data(rand_idx), clust_idx);
    }
#else
    //[ 5, 37, 33])
    // Testing for iris with k = 3
    assert(k == 3);
    cltrs->set_mean(get_thd_data(5),0);
    cltrs->set_mean(get_thd_data(37),1);
    cltrs->set_mean(get_thd_data(33),2);
#endif
}

void kmeans_task_coordinator::set_task_data_ptrs() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it) {
        prune::kmeans_task_thread::ptr thd = std::static_pointer_cast
            <prune::kmeans_task_thread>(*it);
        thd->get_task_queue()->set_data_ptr(thd->get_local_data());
    }
}

double kmeans_task_coordinator::compute_cluster_energy() {
    assert(dist_v.size() == nrow);
    double cluster_energy = std::accumulate(dist_v.begin(),
            dist_v.end(), 0.0);
    assert(cluster_energy < std::numeric_limits<double>::max());
    return cluster_energy;
}

void kmeans_task_coordinator::reinit() {
    std::fill(&dist_v[0], &dist_v[nrow], std::numeric_limits<double>::max());
    cluster_assignments.assign(nrow, base::INVALID_CLUSTER_ID);
    cluster_assignment_counts.assign(k, 0);
    cltrs->clear();
    run_init();
}

void kmeans_task_coordinator::tally_assignment_counts() {
    cluster_assignment_counts.assign(k, 0);
    for (size_t row = 0; row < nrow; row++) {
        assert(cluster_assignments[row] != base::INVALID_CLUSTER_ID
                && cluster_assignments[row] >= 0
                && cluster_assignments[row] < k);
        cluster_assignment_counts[cluster_assignments[row]]++;
    }

    assert((size_t)std::accumulate(cluster_assignment_counts.begin(),
                cluster_assignment_counts.end(), 0) == nrow);
}

kbase::cluster_t kmeans_task_coordinator::dump_state() {
    return kbase::cluster_t(this->nrow, this->ncol, 0, this->k,
            &cluster_assignments[0], &cluster_assignment_counts[0],
            cltrs->get_means());
}

kbase::cluster_t kmeans_task_coordinator::mb_run(double* allocd_data) {
#ifdef PROFILER
    ProfilerStart("mb_kmeans_task_coordinator.perf");
#endif

    if ((double)mb_size / nthreads < 1)
        mb_size = 1;

    // First set the thread mini-batch size
    double mb_perctg = (double)mb_size / nrow;

    for (auto const& th : threads) {
        std::static_pointer_cast<kmeans_task_thread>(th)->
            set_mb_perctg(mb_perctg);
    }

    set_global_ptrs();

    if (NULL == allocd_data) {
        wake4run(ALLOC_DATA);
        wait4complete();
    } else {
        set_thread_data_ptr(allocd_data);
    }

    struct timeval start, end;
    gettimeofday(&start , NULL);
    run_init(); // Initialize clusters

    bool converged = false;
    size_t iter = 0;

    ////////////////////////////////////////////////////////////////////////////
    for (; iter < max_iters; iter++) {
#ifndef BIND
        printf("E-step Iteration: %lu\n", iter);
#endif
        cltrs->set_prev_means();

        wake4run(MB_EM);
        wait4complete();
        mb_iteration_end();

        std::vector<double> diff;
        assert(k*ncol == cltrs->get_means().size());

        kbase::linalg::vdiff(&cltrs->get_means()[0],
                &cltrs->get_prev_means()[0], cltrs->get_means().size(), diff);

        double frob_norm = kbase::linalg::frobenius_norm<double>(
                &diff[0], diff.size());

        if (frob_norm < tolerance) {
            converged = true;
            break;
        }
    }

#ifdef PROFILER
    ProfilerStop();
#endif

    if (converged) {
#ifndef BIND
        printf("K-means converged in %lu iterations\n", iter);
#endif
    } else {
#ifndef BIND
        printf("[Warning]: K-means failed to converge in %lu iterations\n",
                iter);
#endif
    }

    // Run regular EM step to assign all samples to a cluster
    for (auto const& th : threads)
        th->set_prune_init(true);
    wake4run(EM);
    wait4complete();

    // Get cluster counts
    cluster_assignment_counts.assign(k, 0);
    for (auto const& cid : cluster_assignments)
        cluster_assignment_counts[cid]++;

    gettimeofday(&end, NULL);
#ifndef BIND
    printf("\n\nAlgorithmic time taken = %.6f sec\n",
        kbase::time_diff(start, end));
    printf("\n******************************************\n");
#endif

#ifndef BIND
    printf("Final cluster counts: \n");
    kbase::print(cluster_assignment_counts);
    printf("\n******************************************\n");
#endif

    return kbase::cluster_t(this->nrow, this->ncol, iter, this->k,
            &cluster_assignments[0], &cluster_assignment_counts[0],
            cltrs->get_means());
}

/**
 * Main driver for kmeans
 */
kbase::cluster_t kmeans_task_coordinator::run(
        double* allocd_data, const bool numa_opt) {
#ifdef PROFILER
    ProfilerStart("kmeans_task_coordinator.perf");
#endif

    set_global_ptrs();

    if (!numa_opt && NULL == allocd_data) {
        wake4run(ALLOC_DATA);
        wait4complete();
    } else if (allocd_data) { // No NUMA opt
        set_thread_data_ptr(allocd_data);
    } else {
        set_task_data_ptrs();
    }

    struct timeval start, end;
    gettimeofday(&start , NULL);
    run_init(); // Initialize clusters

    size_t iter = 0;

    if (max_iters > 0) {
        // Init Engine
#ifndef BIND
        printf("Running init engine:\n");
#endif
        wake4run(EM);
        wait4complete();
        update_clusters(true);
        set_prune_init(false);

        // Run kmeans loop
        iter = 2;
    }

    num_changed = 0;

    bool converged = false;
    while (iter <= max_iters && max_iters > 0) {
#ifndef BIND
        printf("E-step Iteration: %lu\n", iter);
#if KM_TEST
        printf("Main: Computing cluster distance matrix ...\n");
#endif
#endif
        dm->compute_dist(cltrs, ncol);

        wake4run(EM);
        wait4complete();
        update_clusters(false);

#if VERBOSE
#ifndef BIND
        printf("Cluster assignment counts: \n");
        kbase::print(cluster_assignment_counts);
#endif
#endif

        if (num_changed == 0 ||
                ((num_changed/(double)nrow)) <= tolerance) {
            converged = true;
            break;
        } else {
            num_changed = 0;
        }
        iter++;
    }

    if (iter == 0 && _init_t == kbase::init_t::PLUSPLUS)
        tally_assignment_counts();

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
        printf("[Warning]: K-means failed to converge in %lu iterations\n",
                iter);
#endif
    }

#ifndef BIND
    printf("Final cluster counts: \n");
    kbase::print(cluster_assignment_counts);
    printf("\n******************************************\n");
#endif

    return kbase::cluster_t(this->nrow, this->ncol, iter, this->k,
            &cluster_assignments[0], &cluster_assignment_counts[0],
            cltrs->get_means());
}
} } // End namespace knor, prune

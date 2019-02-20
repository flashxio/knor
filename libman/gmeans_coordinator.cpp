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

#include <random>
#include <stdexcept>

#include "gmeans_coordinator.hpp"
#include "gmeans.hpp"

#include "io.hpp"
#include "clusters.hpp"
#include "linalg.hpp"
#include "AndersonDarling.hpp"
#include "hclust_id_generator.hpp"
#include "thd_safe_bool_vector.hpp"

namespace knor {

gmeans_coordinator::gmeans_coordinator(const std::string fn,
        const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const base::init_t it,
        const double tolerance, const base::dist_t dt,
        const unsigned min_clust_size, const short strictness) :
    xmeans_coordinator(fn, nrow, ncol, k, max_iters, nnodes, nthreads,
            centers, it, tolerance, dt, min_clust_size),
            strictness(strictness) {
}

void gmeans_coordinator::deactivate(const unsigned id) {
    cltr_active_vec->check_set(id, false);
}

void gmeans_coordinator::activate(const unsigned id) {
    cltr_active_vec->check_set(id, true);
    if (id > 0)
        curr_nclust++;
}

void gmeans_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;

    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(gmeans::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, k, hcltrs, &cluster_assignments[0], fn,
                    _dist_t, cltr_active_vec, partition_dist, nearest_cdist,
                    compute_pdist));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
        std::static_pointer_cast<gmeans>(threads[thd_id])
                    ->set_part_id(&part_id[0]);
        std::static_pointer_cast<gmeans>(threads[thd_id])
                    ->set_g_clusters(cltrs);
    }
}

void gmeans_coordinator::assemble_ad_vecs(std::unordered_map<unsigned,
        std::vector<double>>& ad_vecs) {
    for (size_t i = 0; i < nearest_cdist.size(); i++) {
        ad_vecs[part_id[i]].push_back(nearest_cdist[i]);
    }
}

void gmeans_coordinator::compute_ad_stats(
        std::unordered_map<unsigned, std::vector<double>>& ad_vecs) {

    std::vector<unsigned> keys;
    for (auto const& kv : ad_vecs)
        keys.push_back(kv.first);

    std::vector<double> scores(keys.size());

#pragma omp parallel for
    for (size_t idx = 0; idx < keys.size(); idx++) {
        double score = base::AndersonDarling::compute_statistic(
                ad_vecs[keys[idx]].size(), &(ad_vecs[keys[idx]][0]));
        scores[idx] = score;
    }

    // NOTE: We push the score onto the back
    for (size_t idx = 0; idx < keys.size(); idx++)
        ad_vecs[keys[idx]].push_back(scores[idx]);
}

// NOTE: This modifies hcltrs
void gmeans_coordinator::partition_decision() {
    // Each Anderson Darling (AD) vector represents the vector for which
    //  each cluster gets its AD statistic.
    std::unordered_map<unsigned, std::vector<double>> ad_vecs;
    std::vector<double> critical_values;

    // Populate the AD vectors
    assemble_ad_vecs(ad_vecs);

    for (auto& kv : ad_vecs) {
        base::linalg::scale(&(kv.second)[0], kv.second.size());
    }

    // Compute Critical values
    base::AndersonDarling::compute_critical_values(
            ad_vecs.size(), critical_values);

    // Compute AD statistics
    compute_ad_stats(ad_vecs);

    std::vector<size_t> keys;

    hcltrs.get_keys(keys);
    std::vector<bool> revert_cache;
    auto max_pid = (*std::max_element(keys.begin(), keys.end()));
    revert_cache.assign(max_pid+1, false);

    // NOTE: We use ad_vecs.back() to store the score
    for (size_t i = 0; i < keys.size(); i++) {
        unsigned pid = keys[i];
        auto score = ad_vecs[pid].back();

        if (score <= critical_values[strictness]) {
            unsigned lid = hcltrs[pid]->get_zeroid();
            unsigned rid = hcltrs[pid]->get_oneid();

            // Deactivate both lid and rid
            deactivate(lid); deactivate(rid);
            // Deactivate pid
            deactivate(pid);

            revert_cache[pid] = true;
            hcltrs.erase(pid);
            // We can reuse these children ids
            ider->reclaim_id(lid);
            ider->reclaim_id(rid);
            curr_nclust -= 2;

            final_centroids[pid] = std::vector<double>(
                    cltrs->get_mean_rawptr(pid),
                    cltrs->get_mean_rawptr(pid) + ncol);
            cluster_assignment_counts[pid] =
                cluster_assignment_counts[lid] + cluster_assignment_counts[rid];
            cluster_assignment_counts[lid] = cluster_assignment_counts[rid] = 0;
        }
    }

    // Assemble cluster membership
    // TODO: Use dynamic or guided scheduler
#ifdef _OPENMP
#pragma omp parallel for default(shared) firstprivate(revert_cache)
#endif
    for (size_t rid = 0; rid < nrow; rid++) {
        auto pid = part_id[rid];
        // Ignore keys outside in hcltrs
        if (pid < revert_cache.size() && revert_cache[pid]) {
                // Means that row assignments needs to be reverted to part_id
                cluster_assignments[rid] = pid;
        }
    }
}

void gmeans_coordinator::compute_cluster_diffs() {
    auto itr = hcltrs.get_iterator();
    while (itr.has_next()) {
        auto kv = itr.next();
        auto c = std::static_pointer_cast<base::h_clusters>(kv.second);
        c->metadata.resize(ncol+1); // +1th index stores the divisor

        // Compute difference
        for (size_t i = 0; i < ncol; i++)
            base::linalg::vdiff(c->get_mean_rawptr(0),
                    c->get_mean_rawptr(1), ncol, c->metadata);

        // Compute v.dot(v)
        c->metadata[ncol] = base::linalg::dot(&c->metadata[0],
                &c->metadata[0], ncol); // NOTE: last element intentionally ignored
    }
}

// Main driver
base::cluster_t gmeans_coordinator::run(
        double* allocd_data, const bool numa_opt) {
#ifdef PROFILER
    ProfilerStart("gmeans_coordinator.perf");
#endif

    build_thread_state();

    if (!numa_opt && NULL == allocd_data) {
        wake4run(ALLOC_DATA);
        wait4complete();
    } else if (allocd_data) { // No NUMA opt
        set_thread_data_ptr(allocd_data);
    }

    struct timeval start, end;
    gettimeofday(&start , NULL);

    run_hinit(); // Initialize clusters

    // Run loop
    size_t iter = 0;

    while (true) {
        // TODO: Do this simultaneously with H_EM step
        wake4run(MEAN);
        wait4complete();
        combine_partition_means();
        compute_pdist = true;

        for (iter = 0; iter < max_iters; iter++) {
#ifndef BIND
            printf("\nNCLUST: %lu, Iteration: %lu\n", curr_nclust, iter);
#endif
            // Now pick between the cluster splits
            wake4run(H_EM);
            wait4complete();
            update_clusters();
#ifndef BIND
            printf("\nAssignment counts:\n");
            base::sparse_print(cluster_assignment_counts);
            printf("\n*****************************************************\n");
#endif
            if (compute_pdist)
                compute_pdist = false;
        }

        compute_cluster_diffs();

        wake4run(H_SPLIT);
        wait4complete();

        // Decide on split or not here
        partition_decision();

        if (at_cluster_cap()) {
#ifndef BIND
            printf("\n\nCLUSTER SIZE EXIT @ %lu!\n", curr_nclust);
#endif
            break;
        }

        // Update global state
        init_splits(); // Initialize possible splits

        // Break when clusters are inactive due to size
        if (hcltrs.keyless()) {
            assert(steady_state()); // NOTE: Comment when benchmarking
#ifndef BIND
            printf("\n\nSTEADY STATE EXIT!\n");
#endif
            break;
        }
    }
    complete_final_centroids();

#ifdef PROFILER
    ProfilerStop();
#endif

    gettimeofday(&end, NULL);
#ifndef BIND
    printf("\n\nAlgorithmic time taken = %.6f sec\n",
        base::time_diff(start, end));
    printf("\n******************************************\n");
    verify_consistency();
#endif

#ifndef BIND
    printf("Final cluster counts: \n");
    base::sparse_print(cluster_assignment_counts);
    //printf("Final centroids\n");
    //for (auto const& kv : final_centroids) {
        //printf("k: %u, v: ", kv.first); base::print(kv.second);
    //}
    printf("\n******************************************\n");
#endif

    return base::cluster_t(this->nrow, this->ncol, iter,
            cluster_assignments, cluster_assignment_counts,
            final_centroids);
}
} // End namespace knor


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

void gmeans_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;

    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(xmeans::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, k, hcltrs, &cluster_assignments[0], fn,
                    _dist_t, cltr_active_vec, partition_dist, nearest_cdist,
                    compute_pdist));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
        std::static_pointer_cast<xmeans>(threads[thd_id])
                    ->set_part_id(&part_id[0]);
        std::static_pointer_cast<xmeans>(threads[thd_id])
                    ->set_g_clusters(cltrs);
    }
}

void gmeans_coordinator::assemble_ad_stats(std::unordered_map<unsigned,
        std::vector<double>>& ad_vecs) {
    assert(nearest_cdist.size() == part_id.size());
    for (size_t i = 0; i < nearest_cdist.size(); i++) {
        ad_vecs[part_id[i]].push_back(nearest_cdist[i]);

        assert(hcltrs.has_key(part_id[i])); // TODO: RM
    }
}

void gmeans_coordinator::compute_ad_stats(
        std::unordered_map<unsigned, std::vector<double>>& ad_vecs) {
    for (auto& kv : ad_vecs) {
         double score = base::AndersonDarling::compute_statistic(ncol,
                 &kv.second[0]);
         kv.second.push_back(score); // NOTE: We push the score onto the back
    }
}

// NOTE: This modifies hcltrs
void gmeans_coordinator::partition_decision() {
    // Each Anderson Darling (AD) vector represents the vector for which each cluster
    //  gets its AD statistic.
    std::unordered_map<unsigned, std::vector<double>> ad_vecs;
    std::vector<double> critical_values;

    // Populate the AD vectors
    assemble_ad_stats(ad_vecs);

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

    // TODO: RM
    assert(keys.size() == ad_vecs.size());
    for (auto key : keys)
        assert(!(ad_vecs.find(key) == ad_vecs.end()));
    // END TODO: RM

    // NOTE: We use ad_vecs.back() to store the score

    std::vector<bool> remove_cache;
    remove_cache.assign(false, keys.size());

//#pragma omp parallel for
    for (size_t i = 0; i < keys.size(); i++) {
        unsigned pid = keys[i];
        auto score = ad_vecs[pid].back();

        if (score <= critical_values[strictness]) {
#if 1
            printf("\nPart: %u will NOT split! score: %.4f <= crit val: %.4f\n",
                    pid, score, critical_values[strictness]);
#endif
            unsigned lid = hcltrs[pid]->get_zeroid();
            unsigned rid = hcltrs[pid]->get_oneid();

            // FIXME: Move all in children clusters to (parent) partition
            //auto const& lmembers = memb_cltrs[score.lid];
            //for (size_t i = 0; i < memb_cltrs[score.lid].size(); i++)
               //cluster_assignments[lmembers[i]] = score.pid;

            //auto const& rmembers = memb_cltrs[score.lid];
            //for (size_t i = 0; i < memb_cltrs[score.rid].size(); i++)
               //cluster_assignments[rmembers[i]] = score.pid;

            // Deactivate both lid and rid
            deactivate(lid); deactivate(rid);

            // Deactivate pid
            deactivate(pid);
            remove_cache[i] = true;
            final_centroids[pid] = std::vector<double>(
                    cltrs->get_mean_rawptr(pid),
                    cltrs->get_mean_rawptr(pid) + ncol);
        } else {
#if 1
            printf("\nPart: %u will split! score: %.4f > crit val: %.4f\n",
                    pid, score, critical_values[strictness]);
#endif
        }
    }

    for (size_t i = 0; i < remove_cache.size(); i++) {
        if (remove_cache[i])
            hcltrs.erase(keys[i]);
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

    if (_init_t == kbase::init_t::NONE)
        _init_t = kbase::init_t::FORGY;
    run_hinit(); // Initialize clusters

    // Run loop
    size_t iter = 0;

    unsigned curr_nclust = 1;
#if 1
    while (true) {
        printf("Running a Partition Mean step...\n");
        // TODO: Do this simultaneously with H_EM step
        wake4run(MEAN);
        wait4complete();
        combine_partition_means();
        compute_pdist = true;

        for (iter = 0; iter < max_iters; iter++) {
            printf("\n\nNCLUST: %u, Iteration: %lu\n", curr_nclust, iter);
            // Now pick between the cluster splits
            wake4run(H_EM);
            wait4complete();
            update_clusters();
#if 1
            printf("\nAfter update_clusters ... Global hcltrs:\n");
            print_clusters();
            printf("\nAssignment counts:\n");
            base::sparse_print(cluster_assignment_counts);
            //printf("\nAssignments:\n");
            //base::print(cluster_assignments, nrow);
#endif
            printf("\n*****************************************************\n");
            if (compute_pdist)
                compute_pdist = false;
        }

        compute_cluster_diffs();

        wake4run(H_SPLIT);
        wait4complete();

        // Decide on split or not here
        partition_decision();

        if (curr_nclust >= k*2) {
            printf("\n\nCLUSTER SIZE EXIT @ %u!\n", curr_nclust);
            break;
        }

        // Update global state
        init_splits(); // Initialize possible splits

        // Break when clusters are inactive due to size
        if (hcltrs.keyless()) {
            assert(steady_state()); // NOTE: Comment when benchmarking
            printf("\n\nSTEADY STATE EXIT!\n");
            break;
        }
        curr_nclust = hcltrs.keycount()*2 + final_centroids.size();
    }
#ifdef PROFILER
    ProfilerStop();
#endif

#endif

    gettimeofday(&end, NULL);
#ifndef BIND
    printf("\n\nAlgorithmic time taken = %.6f sec\n",
        base::time_diff(start, end));
    printf("\n******************************************\n");
#endif

#ifndef BIND
    printf("Final cluster counts: \n");
    accumulate_cluster_counts();
    base::sparse_print(cluster_assignment_counts);
    printf("\n******************************************\n");
#endif

#if 0
    return base::cluster_t(this->nrow, this->ncol, iter, this->k,
            &cluster_assignments[0], &cluster_assignment_counts[0],
            hcltrs->get_means());
#else
    return base::cluster_t(); // TODO
#endif
}
}

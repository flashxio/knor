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

#include "xmeans_coordinator.hpp"

#include "xmeans.hpp"

#include "io.hpp"
#include "clusters.hpp"
#include "hclust_id_generator.hpp"

namespace knor {

void xmeans_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;

    // TODO: Things that should be in the ctor
    partition_dist.resize(nrow);
    nearest_cdist.resize(nrow);
    // TODO: k can be non 2^n
    cltrs = kbase::clusters::create(base::get_max_hnodes(k), ncol);
    // TODO: End Things that should be in the ctor

    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(xmeans::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, k, &hcltrs, &cluster_assignments[0], fn,
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

void xmeans_coordinator::combine_partition_means() {
    cltrs->unfinalize_all();

    for (auto const& th : threads) {
        auto thd_lcltrs = th->get_local_clusters();
        cltrs->peq(thd_lcltrs);
    }
    cltrs->finalize_all();
}

 /**
   This function computes BIC for parent and children
 We use:
     cltrs -- parent cluster values
     hcltrs -- children values
     nearest_cdist -- data point-assigned cluster dist
     partition dist -- data point-partition dist
 */
void xmeans_coordinator::bic(split_score_t& score,
        std::unordered_map<unsigned, std::vector<unsigned>>& memb_cltrs) {

    // Compute for parent
    double pK = 1; // parent K
    double cK = 2; // children K
    double N = cluster_assignment_counts[score.lid]
                        + cluster_assignment_counts[score.rid];
    double psigma = 0;
    double csigma = 0;

    for (auto const& idx : memb_cltrs[score.lid]) {
            psigma += partition_dist[idx];
            csigma += nearest_cdist[idx];
    }

    for (auto const& idx : memb_cltrs[score.rid]) {
            psigma += partition_dist[idx];
            csigma += nearest_cdist[idx];
    }

    // Parent
    if (N - pK > 0) {
        psigma /= (double) (N - pK);
        double p = (pK - 1) + ncol * pK + 1;

        /* splitting criterion */
        double L = N * std::log(N) - N * std::log(N) - N *
            std::log(2.0 * PI) * .5 - N * ncol *
            std::log(psigma) * .5 - (N - pK) * .5;

        score.pscore = L - p * 0.5 * std::log(N);
    }

    // Children
    if (N - cK > 0) {
        csigma /= (double) (N - cK);
        double p = (cK - 1) + ncol * cK + 1;

        /* splitting criterion */
        double nl = cluster_assignment_counts[score.lid];
        double L = N * std::log(N) - nl * std::log(N) - N *
            std::log(2.0 * PI) * .5 - N * ncol *
            std::log(csigma) * .5 - (N - cK) * .5;

        score.cscore = L - p * 0.5 * std::log(N);

        double nr = cluster_assignment_counts[score.lid];
        L = N * std::log(N) - nr * std::log(N) - N *
            std::log(2.0 * PI) * .5- N * ncol *
            std::log(csigma) * .5 - (N - cK) * .5;
        score.cscore += L - p * .5 * std::log(N);
    }
}

void xmeans_coordinator::compute_bic_scores(
        std::vector<split_score_t>& bic_scores,
    std::unordered_map<unsigned, std::vector<unsigned>>& memb_cltrs) {

    // Creates structures to store bic scores & cluster membership
    auto itr = hcltrs.get_iterator();
    while (itr.has_next()) {
        auto kv = itr.next();
        assert(kv.first == kv.second->get_id());
#if 1
        printf("BIC evaluation for pid: %lu, lid: %u, rid: %u\n",
                kv.first, kv.second->get_zeroid(), kv.second->get_oneid());
#endif
        bic_scores.push_back(split_score_t(kv.second->get_id(),
                    kv.second->get_zeroid(), kv.second->get_oneid()));
        memb_cltrs[kv.first] = std::vector<unsigned>();
    }

#if 0
    assert(std::accumulate(cluster_assignment_counts.begin(),
                cluster_assignment_counts.end(), 0)
                        == static_cast<llong_t>(nrow));
#endif

    // TODO: Slow
    // Creates structures to hold cluster membership
    accumulate(cluster_assignments, memb_cltrs);

    // Computes the bic scores for each parent, child combo
#pragma omp parallel for shared (bic_scores) schedule (dynamic)
    for (size_t idx = 0; idx < bic_scores.size(); idx++) {
        bic(bic_scores[idx], memb_cltrs);
    }
}

// NOTE: This modifies hcltrs
void xmeans_coordinator::partition_decision() {
    std::vector<split_score_t> bic_scores;
    std::unordered_map<unsigned, std::vector<unsigned>> memb_cltrs; // Parent
    compute_bic_scores(bic_scores, memb_cltrs);

    std::vector<bool> remove_cache;
    remove_cache.assign(false, bic_scores.size());

#pragma omp parallel for shared (bic_scores)
    for (size_t i = 0; i < bic_scores.size(); i++) {
        auto score = bic_scores[i];
        if (score.pscore > score.cscore) {
#if 1
            printf("\nPart: %u will NOT split! pscore: %.4f > cscore: %.4f\n",
                    score.pid, score.pscore, score.cscore);
#endif
            // Move all in children clusters to (parent) partition
            auto const& lmembers = memb_cltrs[score.lid];
            for (size_t i = 0; i < memb_cltrs[score.lid].size(); i++)
               cluster_assignments[lmembers[i]] = score.pid;

            auto const& rmembers = memb_cltrs[score.lid];
            for (size_t i = 0; i < memb_cltrs[score.rid].size(); i++)
               cluster_assignments[rmembers[i]] = score.pid;

            // Deactivate both lid and rid
            deactivate(score.lid); deactivate(score.rid);

            // Deactivate pid
            deactivate(score.pid);
            final_centroids[score.pid] = std::vector<double>(
                    cltrs->get_mean_rawptr(score.pid),
                    cltrs->get_mean_rawptr(score.pid) + ncol);
        } else {
#if 1
            printf("\nPart: %u will split! pscore: %.4f <= cscore: %.4f\n",
                    score.pid, score.pscore, score.cscore);
#endif
        }
    }

    for (size_t i = 0; i < remove_cache.size(); i++) {
        if (remove_cache[i])
            hcltrs.erase(bic_scores[i].pid);
    }
}

// Main driver
base::cluster_t xmeans_coordinator::run(
        double* allocd_data, const bool numa_opt) {
#ifdef PROFILER
    ProfilerStart("xmeans_coordinator.perf");
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

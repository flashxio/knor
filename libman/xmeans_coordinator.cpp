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
    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(xmeans::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, k, &hcltrs, &cluster_assignments[0], fn,
                    _dist_t, cltr_active_vec));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
        std::static_pointer_cast<xmeans>(threads[thd_id])
                    ->set_part_id(&part_id[0]);
    }
}

void xmeans_coordinator::spawn(const unsigned& zeroid,
        const unsigned& oneid, const c_part& cp) {

    auto zero_child_ids = ider->get_split_ids(zeroid);
    auto one_child_ids = ider->get_split_ids(oneid);

    // Add parent with two children
    if (cp.l_splittable()) {
        hcltrs[zeroid] = base::h_clusters::create(2, ncol, zeroid,
                zero_child_ids.first, zero_child_ids.second);
        hcltrs[zeroid]->set_mean(get_thd_data(cp.l0), 0);
        hcltrs[zeroid]->set_mean(get_thd_data(cp.l1), 1);
        activate(zero_child_ids.first);
        activate(zero_child_ids.second);
    }

    if (cp.r_splittable()) {
        hcltrs[oneid] = base::h_clusters::create(2, ncol, oneid,
                one_child_ids.first, one_child_ids.second);
        hcltrs[oneid]->set_mean(get_thd_data(cp.r0), 0);
        hcltrs[oneid]->set_mean(get_thd_data(cp.r1), 1);
        activate(one_child_ids.first);
        activate(one_child_ids.second);
    }
}

void xmeans_coordinator::update_clusters() {
    // clear nchanged & means
    nchanged.assign(max_nodes, 0);

    auto itr = hcltrs.get_iterator();
    while (itr.has_next()) {
        auto kv = itr.next();
        if (!kv.second->has_converged())
            kv.second->clear();
    }

    // Serial aggregate of nthread vectors
    for (auto const& thd : threads) {
        // Update the changed cluster count
        auto thd_nchanged =
            (std::static_pointer_cast<xmeans>(thd))->get_nchanged();
        for (size_t i = 0; i < thd_nchanged.size(); i++)
            nchanged[i] += thd_nchanged[i];

        // Update the global hcltrs with local ones
        auto itr = (std::static_pointer_cast<xmeans>(
                        thd))->get_local_hcltrs().get_iterator();
        while (itr.has_next()) {
            auto kv = itr.next();
            hcltrs[kv.first]->peq(kv.second);
        }
    }

    cluster_assignment_counts.assign(max_nodes, 0);

    auto _itr = hcltrs.get_iterator();
    while (_itr.has_next()) {
        auto kv = _itr.next();
        // There are only ever 2 of these for hierarchical algs
        auto pid = kv.first; // partition ID
        auto c = kv.second;
        auto part_nmembers = c->get_num_members(0) + c->get_num_members(1);
        cluster_assignment_counts[c->get_zeroid()] = c->get_num_members(0);
        cluster_assignment_counts[c->get_oneid()] = c->get_num_members(1);

         // Skip clusters that have converged, but are active
        if (!c->has_converged()) {
            c->finalize(0);
            c->finalize(1);

            // Premature End of computation
            if (nchanged[pid] == 0 ||
                    (nchanged[pid]/(double)part_nmembers) <= tolerance) {
                c->set_converged();
            }
        }
    }

#if 0 // Testing
    size_t total_changed = 0;
    for (auto const& val : nchanged)
        total_changed += val;

    printf("Total nchanged: %lu\n", total_changed);
    assert(total_changed <= nrow);
#endif
}

// Main driver
base::cluster_t xmeans_coordinator::run(
        double* allocd_data, const bool numa_opt) {
#ifdef PROFILER
    ProfilerStart("hclust_coordinator.perf");
#endif

    build_thread_state();

    if (!numa_opt && NULL == allocd_data) {
        wake4run(ALLOC_DATA);
        wait4complete();
    } else if (allocd_data) { // No NUMA opt
        set_thread_data_ptr(allocd_data);
    } // Do nothing for numa_opt .. done in binding/knori.hpp

    struct timeval start, end;
    gettimeofday(&start , NULL);

    run_hinit(); // Initialize clusters
    if (_init_t == kbase::init_t::NONE)
        _init_t = kbase::init_t::FORGY;

    printf("After initial init: \n");
    hcltrs[0]->print_means();

    // Run loop
    size_t iter = 0;

    unsigned curr_nclust = 1;
    while (true) {
        for (iter = 0; iter < max_iters; iter++) {
            printf("\n\nNCLUST: %u, Iteration: %lu\n", curr_nclust, iter);
            // Now pick between the cluster splits
            wake4run(H_EM);
            wait4complete();

            update_clusters();
#if 0
            printf("\nAfter update_clusters ... Global hcltrs:\n");
            print_active_clusters();
#endif
            printf("\n*****************************************************\n");
        }

        curr_nclust*=2;
        if (curr_nclust > k)
            break;

        // Update global state
        init_splits(); // Initialize possible splits
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

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

#include "hclust_coordinator.hpp"
#include "hclust.hpp"
#include "io.hpp"
#include "clusters.hpp"
#include "hclust_id_generator.hpp"

namespace knor {

hclust_coordinator::hclust_coordinator(const std::string fn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const base::init_t it,
        const double tolerance, const base::dist_t dt,
        const unsigned min_clust_size) :
    coordinator(fn, nrow, ncol, (base::get_hclust_floor(k)/2), max_iters,
            nnodes, nthreads, centers, it, tolerance, dt) {

        cltr_active_vec.assign(k, false);
        activate(0);
        this->min_clust_size = min_clust_size;

        ui_distribution = std::uniform_int_distribution<unsigned>(0, nrow-1);

        max_nodes = base::get_max_hnodes(k*2);
        hcltrs.set_max_capacity(max_nodes);

        if (centers) {
            // There must be at least one
            hcltrs[0] = base::h_clusters::create(2, ncol, centers);
        } else {
            hcltrs[0] = base::h_clusters::create(2, ncol);
        }
        hcltrs[0]->set_id(0);

        std::fill(cluster_assignments.begin(), cluster_assignments.end(), 0);
        part_id.assign(nrow, 0);

        nchanged.assign(max_nodes, 0);
        cluster_assignment_counts.assign(max_nodes, 0);
        ider = hclust_id_generator::create();
    }

void hclust_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;
    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(hclust::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, k, &hcltrs, &cluster_assignments[0], fn, _dist_t,
                    cltr_active_vec));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
        std::static_pointer_cast<hclust>(threads[thd_id])
                    ->set_part_id(&part_id[0]);
    }
}

void hclust_coordinator::partition_mean(base::vmap<
        std::shared_ptr<base::clusters>>& part_hcltrs) {
    wake4run(MEAN);
    wait4complete();

    if (part_hcltrs.empty())
        part_hcltrs.set_max_capacity(max_nodes);
    else
        part_hcltrs.clear();

    for (auto const& thd : threads) {
        // Update the global hcltrs with local ones
        auto itr = (std::static_pointer_cast<hclust>(
                        thd))->get_local_hcltrs().get_iterator();

        while (itr.has_next()) {
            auto kv = itr.next();
            part_hcltrs[kv.first]->peq(kv.second);
        }
    }

    // TODO: Verify correctness
    for (size_t i = 0; i < part_hcltrs.size(); i++) {
        if (part_hcltrs.has_key(i))
            part_hcltrs[i]->finalize_all();
    }
}

/**
  * Require a method by which to pick centroids when samples in a cluster are
  *     not contiguously numbered
  */
unsigned hclust_coordinator::forgy_select(const unsigned cid) {
    _mutex.lock(); // We need these ordered for a determinant result
    unsigned rand_idx = ui_distribution(ui_generator);
    _mutex.unlock();
    const long max_rid = nrow - 1;

    // The random item we picked is indeed in the cluster we're evaluating
    if (cluster_assignments[rand_idx] == cid)
        return rand_idx;

    // Else walk the cluster assignments until you find and row that is
    long cnt = static_cast<long>(rand_idx);
    // Go forward first
    while (++cnt < max_rid)
        if (cluster_assignments[cnt] == cid)
            return cnt;

    cnt = static_cast<long>(rand_idx);
    // Go backward
    while (--cnt < max_rid)
        if (cluster_assignments[cnt] == cid)
            return cnt;

    throw std::runtime_error((std::string("No samples in cluster ")
                + std::to_string(cid)).c_str());
}

void hclust_coordinator::forgy_init() {
        auto splits = ider->get_split_ids(0);
        auto cluster_ptr = hcltrs[0];

        auto rand_idx = ui_distribution(ui_generator);
        cluster_ptr->set_mean(get_thd_data(
                    rand_idx), 0);
        cluster_ptr->set_zeroid(splits.first);
        activate(splits.first);

        rand_idx = ui_distribution(ui_generator);
        cluster_ptr->set_mean(get_thd_data(
                    rand_idx), 1);
        cluster_ptr->set_oneid(splits.second);
        activate(splits.second);
}

void hclust_coordinator::run_hinit() {
    switch(_init_t) {
        case kbase::init_t::FORGY:
            forgy_init();
            break;
        case kbase::init_t::NONE:
            break;
        default:
            throw std::runtime_error("Unknown initialization type");
    }
}

void hclust_coordinator::print_clusters() {
    auto itr = hcltrs.get_iterator();
    while (itr.has_next()) {
        auto kv = itr.next();
        printf("cid: %lu\n", kv.first);
        kv.second->print_means();
    }
}

// How to initialize when splitting
void hclust_coordinator::inner_init(std::vector<unsigned>& remove_cache) {
    // TODO: ||ize
    std::vector<size_t> ids;
    hcltrs.get_keys(ids);

    for (auto const& id : ids) {
        auto zeroid = hcltrs[id]->get_zeroid();
        auto oneid = hcltrs[id]->get_oneid();

        // l0 & l1 are associated with zeroid
        // r0 & r1 are associated with oneid
        bool l0_complete, l1_complete, r0_complete, r1_complete;
        l0_complete = l1_complete = r0_complete = r1_complete = false;

        // Min cluster deactivation
        if (cluster_assignment_counts[zeroid] < min_clust_size) {
#if 0
            printf("MIN CLUST DEACTIVATION!\n");
#endif
            deactivate(zeroid); // So we no longer process this
            // Record this as an output
            final_centroids[zeroid] = std::vector<double>(
                    hcltrs[id]->get_mean_rawptr(0),
                    hcltrs[id]->get_mean_rawptr(0) + ncol);
            // Don't split
            l0_complete = l1_complete = true;
        }

        if (cluster_assignment_counts[oneid] < min_clust_size) {
#if 0
            printf("MIN CLUST DEACTIVATION!\n");
#endif
            deactivate(oneid); // But never remove
            final_centroids[oneid] = std::vector<double>(
                    hcltrs[id]->get_mean_rawptr(1),
                    hcltrs[id]->get_mean_rawptr(1) + ncol);
            r0_complete = r1_complete = true;
        }

        remove_cache.push_back(id); // Should have no data members
        assert(cluster_assignment_counts[id] == 0);

        c_part cp;

        for (unsigned vid = 0; vid < nrow; vid++) {
            if (l0_complete && l1_complete && r0_complete && r1_complete)
                break; // Out of inner loop

            if (cluster_assignments[vid] == zeroid) {
                if (!l0_complete) {
                    cp.l0 = vid;
                    l0_complete = true;
                } else if (!l1_complete) {
                    cp.l1 = vid;
                    l1_complete = true;

                    // oneid has too few members to split
                    if (!is_active(oneid))
                        break;
                }
            } else if (cluster_assignments[vid] == oneid) {
                if (!r0_complete) {
                    cp.r0 = vid;
                    r0_complete = true;
                } else if (!r1_complete) {
                    cp.r1 = vid;
                    r1_complete = true;

                    // zeroid has too few membs to split
                    if (!is_active(zeroid))
                        break;
                }
            }
        }
        spawn(zeroid, oneid, cp); // This edits hcltrs
    }
}

void hclust_coordinator::spawn(const unsigned& zeroid,
        const unsigned& oneid, const c_part& cp) {
    // Add parent with two children
    if (cp.l_splittable()) { // NOTE: Could do an active check, but redundant
        auto zero_child_ids = ider->get_split_ids(zeroid);
        hcltrs[zeroid] = base::h_clusters::create(2, ncol, zeroid,
                zero_child_ids.first, zero_child_ids.second);
        hcltrs[zeroid]->set_mean(get_thd_data(cp.l0), 0);
        hcltrs[zeroid]->set_mean(get_thd_data(cp.l1), 1);
        activate(zero_child_ids.first);
        activate(zero_child_ids.second);
    }

    if (cp.r_splittable()) {
        auto one_child_ids = ider->get_split_ids(oneid);
        hcltrs[oneid] = base::h_clusters::create(2, ncol, oneid,
                one_child_ids.first, one_child_ids.second);
        hcltrs[oneid]->set_mean(get_thd_data(cp.r0), 0);
        hcltrs[oneid]->set_mean(get_thd_data(cp.r1), 1);
        activate(one_child_ids.first);
        activate(one_child_ids.second);
    }
}

void hclust_coordinator::init_splits() {
    std::vector<unsigned> remove_cache;
    inner_init(remove_cache);

    // Now update with new clusters added and delete parent
    for (unsigned id : remove_cache) {
#if 0
        printf("\t\tErasing: %u\n", id);
#endif
        hcltrs.erase(id); // Delete
        deactivate(id);
    }

    // Update partition ID
    part_id = cluster_assignments;
}

// Helper
void hclust_coordinator::accumulate_cluster_counts() {
    cluster_assignment_counts.assign(max_nodes, 0);

    for (auto const& cid : cluster_assignments) {
            cluster_assignment_counts[cid]++;
    }
}

void hclust_coordinator::deactivate(const unsigned id) {
#if 0
    printf("Deactivating %u\n", id);
#endif
    cltr_active_vec[id] = false;
}

void hclust_coordinator::activate(const unsigned id) {
#if 0
    printf("Activating %u\n", id);
#endif
    cltr_active_vec[id] = true;
}

bool hclust_coordinator::is_active(const unsigned id) {
    return cltr_active_vec[id];
}

// NOTE: Only use after you've tried to split, because there can be no active
//  clusters, but clusters can still be splittable during a run.
const bool hclust_coordinator::steady_state() const {
    for (auto const& flag : cltr_active_vec)
        if (flag)
            return false;
    return true; // No more clusters are active & none can be split
}

void hclust_coordinator::update_clusters() {
    // clear nchanged & means
    nchanged.assign(max_nodes, 0);

    auto itr = hcltrs.get_iterator();
    while (itr.has_next()) {
        auto kv = itr.next();
        assert(is_active(kv.first));
        if (!kv.second->has_converged()) {
            kv.second->clear();
        }
    }

    // Serial aggregate of nthread vectors
    for (auto const& thd : threads) {
        // Update the changed cluster count
        auto thd_nchanged =
            (std::static_pointer_cast<hclust>(thd))->get_nchanged();
        for (size_t i = 0; i < thd_nchanged.size(); i++)
            nchanged[i] += thd_nchanged[i];

        // Update the global hcltrs with local ones
        auto itr = (std::static_pointer_cast<hclust>(
                        thd))->get_local_hcltrs().get_iterator();
        while (itr.has_next()) {
            auto kv = itr.next();
            hcltrs[kv.first]->peq(kv.second);
        }
    }

    auto _itr = hcltrs.get_iterator();
    while (_itr.has_next()) {
        auto kv = _itr.next();
        // There are only ever 2 of these for hclust
        auto pid = kv.first; // partition ID
        auto c = kv.second;
        auto part_nmembers = c->get_num_members(0) + c->get_num_members(1);
        cluster_assignment_counts[kv.first] = 0; // Zero out parent
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

/**
 * Main driver
 */
base::cluster_t hclust_coordinator::run(
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

    if (_init_t == kbase::init_t::NONE)
        _init_t = kbase::init_t::FORGY;
    run_hinit(); // Initialize clusters

#if 1
    printf("After initial init: \n");
    hcltrs[0]->print_means();
#endif

    // Run loop
    size_t iter = 0;

    unsigned curr_nclust = 2;
    while (true) {
        for (iter = 0; iter < max_iters; iter++) {
            printf("\n\nNCLUST: %u, Iteration: %lu\n", curr_nclust, iter);
            // Now pick between the cluster splits
            wake4run(H_EM);
            wait4complete();

            update_clusters();
#if 1
            //printf("\nAfter update_clusters ... Global hcltrs:\n");
            //print_clusters();
            printf("\nAssignment counts:\n");
            base::sparse_print(cluster_assignment_counts);
            //printf("\nAssignments:\n");
            //base::print(cluster_assignments, nrow);
#endif
            printf("\n*****************************************************\n");
        }

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

    // Finally update the final centroids with the values in hcltrs
    // TODO: Time waster
    auto itr = hcltrs.get_iterator();
    while (itr.has_next()) {
        auto const& c = itr.next().second;
        final_centroids[c->get_zeroid()] =
            std::vector<double>(c->get_mean_rawptr(0),
                    c->get_mean_rawptr(0) + ncol);
        final_centroids[c->get_oneid()] =
            std::vector<double>(c->get_mean_rawptr(1),
                    c->get_mean_rawptr(1) + ncol);
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
    printf("Final cluster counts & means: \n");
    //accumulate_cluster_counts();
    base::sparse_print(cluster_assignment_counts);
    //for (auto const& kv : final_centroids) {
        //printf("k: %u, v: ", kv.first); base::print(kv.second);
    //}
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

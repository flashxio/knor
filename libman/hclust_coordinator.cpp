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
static unsigned MIN_CLUST_SIZE = 2; // TODO: Param

hclust_coordinator::hclust_coordinator(const std::string fn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const base::init_t it,
        const double tolerance, const base::dist_t dt) :
    coordinator(fn, nrow, ncol, k/2, max_iters,
            nnodes, nthreads, centers, it, tolerance, dt) {

        cltr_active_vec = new std::vector<bool>(); // We know the max size!
        cltr_active_vec->assign(k, false);
        activate(0);

        ui_distribution = std::uniform_int_distribution<unsigned>(0, nrow-1);

        if (centers) {
            // There must be at least one
            hcltrs[0] = base::h_clusters::create(2, ncol, centers);
        } else {
            hcltrs[0] = base::h_clusters::create(2, ncol);
        }

        std::fill(cluster_assignments.begin(), cluster_assignments.end(), 0);
        part_id.assign(nrow, 0);
        build_thread_state();

        nchanged.assign(k, 0);
    }

void hclust_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;
    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(hclust::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, &hcltrs, &cluster_assignments[0], fn, _dist_t));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
        std::static_pointer_cast<hclust>(threads[thd_id])
                    ->set_cltr_active_vec(cltr_active_vec);
        std::static_pointer_cast<hclust>(threads[thd_id])
                    ->set_ider(get_ider());
        std::static_pointer_cast<hclust>(threads[thd_id])
                    ->set_part_id(&part_id[0]);
    }
}

void hclust_coordinator::kmeanspp_init() {
#if 0
    struct timeval start, end;
    gettimeofday(&start , NULL);

    std::vector<double> dist_v;
    dist_v.assign(nrow, std::numeric_limits<double>::max());
    set_thd_dist_v_ptr(&dist_v[0]);

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, nrow-1);

    // Choose c1 uniformly at random
    unsigned selected_idx = distribution(generator);
    hcltrs->set_mean(get_thd_data(selected_idx), 0);
    dist_v[selected_idx] = 0.0;
    cluster_assignments[selected_idx] = 0;

    unsigned clust_idx = 0; // The number of clusters assigned

    std::uniform_real_distribution<double> ur_distribution(0.0, 1.0);

    // Choose next center c_i with weighted prob
    while (true) {
        set_thread_clust_idx(clust_idx); // Set the current cluster index
        wake4run(KMSPP_INIT); // Run || distance comp to clust_idx
        wait4complete();

        // Sum the per thread cumulative dists
        double cuml_dist = reduction_on_cuml_sum();

        cuml_dist = (cuml_dist * ur_distribution(generator)) / (RAND_MAX - 1.0);
        if (++clust_idx >= k)  // No more centers needed
            break;

        for (size_t row = 0; row < nrow; row++) {
            cuml_dist -= dist_v[row];
            if (cuml_dist <= 0) {
                hcltrs->set_mean(get_thd_data(row), clust_idx);
                cluster_assignments[row] = clust_idx;
                break;
            }
        }
        assert(cuml_dist <= 0);
    }

    gettimeofday(&end, NULL);
#endif
}

void hclust_coordinator::random_partition_init() {
#if 0
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, k-1);

    for (unsigned row = 0; row < nrow; row++) {
        unsigned asgnd_clust = distribution(generator);
        const double* dp = get_thd_data(row);

        hcltrs->add_member(dp, asgnd_clust);
        cluster_assignments[row] = asgnd_clust;
    }

    hcltrs->finalize_all();
#endif
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
        cluster_ptr->set_id(0);

        auto rand_idx = ui_distribution(ui_generator);
        printf("Selected row: %u for cid: %u\n ", rand_idx, splits.first);

        cluster_ptr->set_mean(get_thd_data(
                    rand_idx), 0);
        cluster_ptr->set_zeroid(splits.first);
        activate(splits.first);

        rand_idx = ui_distribution(ui_generator);
        printf("Selected row: %u for cid: %u\n ", rand_idx, splits.second);

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

void hclust_coordinator::print_active_clusters() {
    for (auto kv : hcltrs) {
        printf("CID: %u\n", kv.first);
        kv.second->print_means();
    }
}

// How to initialize when splitting
void hclust_coordinator::inner_init(std::vector<unsigned>& remove_cache) {
    // TODO: ||ize
    // First check for empty clusters
    for (size_t i = 0; i < cluster_assignment_counts.size(); i++) {
        if (cluster_assignment_counts[i] < MIN_CLUST_SIZE)
            deactivate(i);
    }

    printf("\nIn inner_init iterating cluster IDs:\n");
    for (auto kv : hcltrs) {
        printf("cid: %u, zeroid: %u, oneid: %u\n",
                kv.first, kv.second->get_zeroid(), kv.second->get_oneid());
    }

    // TODO: Improve
    std::vector<unsigned> ids;
    for (auto kv : hcltrs) {
        ids.push_back(kv.first);
    }

    for (auto id : ids) {
        printf("Processing parent: %u\n", id);
        auto zeroid = hcltrs[id]->get_zeroid();
        auto oneid = hcltrs[id]->get_oneid();

        bool l0_complete, l1_complete, r0_complete, r1_complete;
        l0_complete = l1_complete = r0_complete = r1_complete = false;

        remove_cache.push_back(id);
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

                    if (!is_active(oneid)) // oneid has too few members to split
                        break;
                }
            } else if (cluster_assignments[vid] == oneid) {
                if (!r0_complete) {
                    cp.r0 = vid;
                    r0_complete = true;
                } else if (!r1_complete) {
                    cp.r1 = vid;
                    r1_complete = true;

                    if (!is_active(zeroid)) // zeroid has too few membs to split
                        break;
                }
            }
        }
        spawn(zeroid, oneid, cp); // This edits hcltrs
    }
}

void hclust_coordinator::spawn(const unsigned& zeroid,
        const unsigned& oneid, const c_part& cp) {

    printf("In spawn\n");
    printf("zeroid: %u, oneid: %u\n", zeroid, oneid);
    cp.print();
    auto zero_child_ids = ider->get_split_ids(zeroid);
    auto one_child_ids = ider->get_split_ids(oneid);

    // Add parent with two children
    if (cp.l_splittable()) {
        hcltrs[zeroid] = base::h_clusters::create(2, ncol, zeroid,
                zero_child_ids.first, zero_child_ids.second);
        hcltrs[zeroid]->set_mean(get_thd_data(cp.l0), 0);
        hcltrs[zeroid]->set_mean(get_thd_data(cp.l1), 1);
        printf("SPAWN for %u --> \n", zeroid);
        activate(zero_child_ids.first);
        activate(zero_child_ids.second);
    }

    if (cp.r_splittable()) {
        hcltrs[oneid] = base::h_clusters::create(2, ncol, oneid,
                one_child_ids.first, one_child_ids.second);
        hcltrs[oneid]->set_mean(get_thd_data(cp.r0), 0);
        hcltrs[oneid]->set_mean(get_thd_data(cp.r1), 1);
        printf("SPAWN for %u --> \n", oneid);
        activate(one_child_ids.first);
        activate(one_child_ids.second);
    }
}

void hclust_coordinator::init_splits() {
#if 1
    printf("Cluster assignments!\n");
    base::print_vector(cluster_assignments, nrow);
    printf("\n\n");
#endif

    std::vector<unsigned> remove_cache;
    inner_init(remove_cache);

    // Now update with new clusters added and delete parent
    for (unsigned id : remove_cache) {
        printf("\t\tErasing: %u\n", id);
        hcltrs.erase(id); // Delete
        deactivate(id);
    }

    printf("hcltrs after init_splits:\n");
    print_active_clusters();

    // Update partition ID
    part_id = cluster_assignments;
#if 1
    // We need a copy
    assert(&part_id[0] != &cluster_assignments[0]);
#endif
}

// Helper
void hclust_coordinator::accumulate_cluster_counts() {
    // We need cluster_assignment_counts to be a map
    cluster_assignment_counts.clear(); // Should be empty when we do this!

    for (auto cid : cluster_assignments) {
        if (cluster_assignment_counts.find(cid)
                != cluster_assignment_counts.end()) {
            cluster_assignment_counts[cid]++;
        } else  {
            cluster_assignment_counts[cid] = 1;
        }
    }
}

void hclust_coordinator::deactivate(const unsigned id) {
#if 1
    printf("\tDeactivating CID: %u\n", id);
#endif
    (*cltr_active_vec)[id] = false;
}

void hclust_coordinator::activate(const unsigned id) {
#if 1
    printf("\tActivating CID: %u\n", id);
#endif
    (*cltr_active_vec)[id] = true;
}

bool hclust_coordinator::is_active(const unsigned id) {
    return (*cltr_active_vec)[id];
}

void hclust_coordinator::update_clusters() {
    // clear nchanged & means
    nchanged.assign(k, 0);
    for (auto kv : hcltrs) {

        // No need to update the state of this partition because it's converged
        if (!kv.second->has_converged())
            kv.second->clear(); // clear means if it's changed
    }

    // Serial aggregate of nthread vectors
    for (auto thd : threads) {
        // Update the changed cluster count
        for (auto kv : (std::static_pointer_cast<hclust>(thd))->get_nchanged())
            nchanged[kv.first] += kv.second;

        // Update the global hcltrs with local ones
        for (auto kv : (std::static_pointer_cast<hclust>(
                        thd))->get_local_hcltrs())
            hcltrs[kv.first]->peq(kv.second);
    }

    cluster_assignment_counts.clear(); // Empty the map

    for (auto kv : hcltrs) {
        // There are only ever 2 of these for hclust
        auto pid = kv.first; // partition ID
        auto c = kv.second;
        auto part_nmembers = c->get_num_members(0) + c->get_num_members(1);
        cluster_assignment_counts[c->get_zeroid()] = c->get_num_members(0);
        cluster_assignment_counts[c->get_oneid()] = c->get_num_members(1);

        if (c->has_converged()) // No need to do anything now
            continue;

        c->finalize(0);
        c->finalize(1);

        // Premature End of computation
        if (nchanged[pid] == 0 ||
                (nchanged[pid]/(double)part_nmembers) <= tolerance) {
            printf("\n\tPID: %u converged!\n", pid);
            c->set_converged();
        }
#if 1
        printf("PID: %u, with mean: ", pid); c->print_means();
#endif
    }

    printf("Cluster assignments:\n");
    base::print_vector(cluster_assignments, nrow);
    printf("\n\n");

#if 1 // Testing
    size_t total_changed = 0;
    for (auto val : nchanged)
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
    ProfilerStart("libman/hclust_coordinator.perf");
#endif

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

    /*TODO: Or all clusters are inactive*/
    //for (unsigned curr_nclust = 1; curr_nclust < k; curr_nclust*=2)
    unsigned curr_nclust = 1;
    while (true) {
        for (iter = 0; iter < max_iters; iter++) {
            printf("\n\nNCLUST: %u, Iteration: %lu\n", curr_nclust, iter);
            // Now pick between the cluster splits
            wake4run(H_EM);
            wait4complete();
#if 1
            // Accumulate the counts
            printf("After H_EM step Global hcltrs: \n");
            print_active_clusters();
            printf("Cluster assignments:\n");
            base::print_vector(cluster_assignments, nrow);
            printf("\n\n");
#endif

            update_clusters();
#if 1
            printf("\nAfter update_clusters ... Global hcltrs:\n");
            print_active_clusters();
            printf("Global cluster assignment counts:\n");
            base::print(cluster_assignment_counts);
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
    base::print(cluster_assignment_counts);
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

std::shared_ptr<hclust_id_generator> hclust_coordinator::get_ider() {
    if (nullptr == ider)
        ider = hclust_id_generator::create();
    return ider;
}

hclust_coordinator::~hclust_coordinator() {
    delete (cltr_active_vec);
}
}

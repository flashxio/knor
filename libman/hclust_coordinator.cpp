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
    coordinator(fn, nrow, ncol, k, max_iters,
            nnodes, nthreads, centers, it, tolerance, dt) {

        cltr_active_vec = new std::vector<bool>(); // We know the max size!
        cltr_active_vec->assign(k, false);
        activate(0);

        ui_distribution = std::uniform_int_distribution<unsigned>(0, nrow-1);

        if (centers) {
            // There must be at least one
            hcltrs[0] = base::h_clusters::create(2, ncol, centers);
            //hcltrs->insert({0, base::h_clusters::create(2, ncol, centers)});
        } else {
            hcltrs[0] = base::h_clusters::create(2, ncol);
            //hcltrs->insert({0, base::h_clusters::create(2, ncol)});
        }

        std::fill(cluster_assignments.begin(), cluster_assignments.end(), 0);
        part_id.assign(nrow, 0);
        build_thread_state();
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
    // TODO: ||ize
    //#ifdef _OPENMP
    //#pragma omp parallel for schedule(dynamic)
    //#endif
    for (size_t i = 0; i < cltr_active_vec->size(); i++) {
        if (is_active(i)) {
            auto cluster_ptr = hcltrs[i];
            //auto cluster_ptr = hcltrs->at(i);

            for (unsigned clust_idx = 0; clust_idx < 2; clust_idx++) {
                unsigned rand_idx = forgy_select(i);
#if 1
                printf("Cluster %lu selected rid: %u for c:%u\n",
                        i, rand_idx, clust_idx);
#endif
                auto splits = ider->get_split_ids(i);

                cluster_ptr->set_mean(get_thd_data(rand_idx), clust_idx);
                cluster_ptr->set_zeroid(splits.first);
                cluster_ptr->set_oneid(splits.second);
            }
        }
    }
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
    for (auto kv : hcltrs) {
        auto zeroid = kv.second->get_zeroid();
        auto oneid = kv.second->get_oneid();

        bool l0_complete, l1_complete, r0_complete, r1_complete;
        l0_complete = l1_complete = r0_complete = r1_complete = false;

        remove_cache.push_back(kv.first);
        c_part cp;

        for (unsigned vid = 0; vid < nrow; vid++) {
            if (l0_complete && l1_complete && r0_complete && r1_complete)
                break; // Out of inner loop

            if (cluster_assignments[vid] == zeroid) {
                if (!l0_complete) {
                    cp.l0 = vid;
#if 1
                    printf("L0: set to rid: %u\n", vid);
#endif
                    l0_complete = true;
                } else if (!r0_complete) {
                    cp.r0 = vid;
#if 1
                    printf("R0: set to rid: %u\n", vid);
#endif
                    r0_complete = true;
                }
            } else if (cluster_assignments[vid] == oneid) {
                if (!l1_complete) {
                    cp.l1 = vid;
#if 1
                    printf("L1: set to rid: %u\n", vid);
#endif
                    l1_complete = true;
                } else if (!r1_complete) {
                    cp.r1 = vid;
#if 1
                    printf("R1: set to rid: %u\n", vid);
#endif
                    r1_complete = true;
                }
            }
        }

        cp.check(); // TODO: RM
        spawn(zeroid, oneid, cp);
    }
}

void hclust_coordinator::spawn(const unsigned& zeroid,
        const unsigned& oneid, const c_part& cp) {
        auto zero_child_ids = ider->get_split_ids(zeroid);
        auto one_child_ids = ider->get_split_ids(oneid);

        // Add parent with two children
        // TODO: memoize these
        hcltrs[zeroid] = base::h_clusters::create(2, ncol);
        hcltrs[zeroid]->set_mean(get_thd_data(cp.l0), 0);
        hcltrs[zeroid]->set_mean(get_thd_data(cp.l1), 1);
        hcltrs[zeroid]->set_zeroid(zero_child_ids.first);
        hcltrs[zeroid]->set_oneid(zero_child_ids.second);
        activate(zeroid);

        hcltrs[oneid] = base::h_clusters::create(2, ncol);
        hcltrs[oneid]->set_mean(get_thd_data(cp.r0), 0);
        hcltrs[oneid]->set_mean(get_thd_data(cp.r1), 1);
        hcltrs[oneid]->set_zeroid(one_child_ids.first);
        hcltrs[oneid]->set_oneid(one_child_ids.second);
        activate(oneid);
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
        hcltrs.erase(id); // Delete
        deactivate(id);
    }

    curr_nclust = hcltrs.size();

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
    for (auto kv : hcltrs) {
        nchanged[kv.second->get_zeroid()] = 0;
        nchanged[kv.second->get_oneid()] = 0;
        kv.second->clear(); // clear means
    }

    // Serial aggregate of nthread vectors
    for (auto thd : threads) {
        // Update the changed cluster count
        for (kv : (std::static_pointer_cast<hclust>(thd))->get_nchanged())
            nchanged[kv.first] += kv.second;

        // Update the global hcltrs with local ones
        for (kv : (std::static_pointer_cast<hclust>(thd))->get_local_hcltrs())
            hcltrs[kv.first]->peq(kv.second);
    }

    cluster_assignment_counts.clear(); // Empty the map

    size_t chk_nmemb = 0; // Global
    for (auto kv : hcltrs) {
        // There are only ever 2 of these for hclust
        kv.second->finalize(0);

        cluster_assignment_counts[kv.second->get_zeroid()] =
            kv.second->get_num_members(0);

        //if (kv.second->get_num_members(0) < MIN_CLUST_SIZE) {
            //compute_skip(kv.first);
            //deactivate(kv.first);
        //}

        kv.second->finalize(1);
        cluster_assignment_counts[kv.second->get_oneid()] =
            kv.second->get_num_members(1);

#if 1
        printf("CID: %u, with mean: ", kv.first);
        kv.second->print_means();
        printf("With num_members[0]: %lu\n", kv.second->get_num_members(0));
        printf("With num_members[1]: %lu\n", kv.second->get_num_members(1));
#endif

#if 1
        chk_nmemb += cluster_assignment_counts[kv.second->get_zeroid()];
        chk_nmemb += cluster_assignment_counts[kv.second->get_oneid()];
#endif
    }

    printf("chk_nmemb: %lu\n", chk_nmemb);
    assert(chk_nmemb == nrow);

#if 1 // Testing
    size_t total_changed = 0;
    for (auto kv : nchanged)
        total_changed += kv.second;

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
    bool converged = false;
    size_t iter = 0;
    curr_nclust = 1;

    while (curr_nclust <= k) { /*TODO: Or all clusters are inactive*/

        for (iter = 0; iter < max_iters; iter++) {
            printf("\n\nNCLUST: %u, Iteration: %lu\n", curr_nclust, iter);
            // Now pick between the cluster splits
            wake4run(H_EM);
            wait4complete();
#if 1
            // Accumulate the counts
            printf("After H_EM step Global hcltrs: \n");
            print_active_clusters();
            printf("Cluster assignments counts:\n");
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
            // TODO: Per cluster tolerance early termination
            //if (num_changed == 0 ||
            //((num_changed/(double)nrow)) <= tolerance) {
            //converged = true;
            //break;
            //}
            printf("\n*****************************************************\n");
        }

        if (curr_nclust == k)
            break;

        // Update global state for curr_nclust
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
    if (converged) {
#ifndef BIND
        printf("HClust converged in %lu iterations\n", iter);
#endif
    } else {
#ifndef BIND
        printf("[Warning]: HClust failed to converge in %lu"
            " iterations\n", iter);
#endif
    }

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
    return base::cluster_t();
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

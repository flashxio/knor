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
        const double tolerance, const base::dist_t dt) :
    coordinator(fn, nrow, ncol, k, max_iters,
            nnodes, nthreads, centers, it, tolerance, dt) {

        cltr_active_vec = new std::vector<bool>(1);
        (*cltr_active_vec)[0] = true;
        ui_distribution = std::uniform_int_distribution<unsigned>(0, nrow-1);

        if (centers) {
            // There must be at least one
            hcltrs[0] = base::h_clusters::create(2, ncol, centers);
            //hcltrs->insert({0, base::h_clusters::create(2, ncol, centers)});
        } else {
            hcltrs[0] = base::h_clusters::create(2, ncol);
            //hcltrs->insert({0, base::h_clusters::create(2, ncol)});
        }

        // Init for 1st clusters
        nchanged[0] = 0;

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
        if ((*cltr_active_vec)[i]) {
            auto cluster_ptr = hcltrs[i];
            //auto cluster_ptr = hcltrs->at(i);

            for (unsigned clust_idx = 0; clust_idx < 2; clust_idx++) {
                unsigned rand_idx = forgy_select(i);
                printf("Cluster %lu selected rid: %u for c:%u\n",
                        i, rand_idx, clust_idx);
                base::print_arr<double>(get_thd_data(rand_idx), ncol);
                printf("\n");

                auto splits = ider->get_split_ids(i);

                cluster_ptr->set_mean(get_thd_data(rand_idx), clust_idx);
                cluster_ptr->set_zeroid(splits.first);
                cluster_ptr->set_oneid(splits.second);
                cluster_ptr->print_means();
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

void hclust_coordinator::init_splits() {
    // TODO: ||ize
    std::vector<unsigned> remove_cache;

    for (auto kv : hcltrs) {
        auto zeroid = kv.second->get_zeroid();
        auto oneid = kv.second->get_oneid();

        bool l0_complete, l1_complete, r0_complete, r1_complete;
        l0_complete = l1_complete = r0_complete = r1_complete = false;

        remove_cache.push_back(kv.first);
        unsigned l0, l1, r0, r1;
        l0 = l1 = r0 = r1 = std::numeric_limits<unsigned>::max();

        for (auto vid : cluster_assignments) {
            if (l0_complete && l1_complete && r0_complete && r1_complete)
                break; // Out of inner loop

            if (vid == zeroid) {
                if (!l0_complete) {
                    l0 = vid;
                    l0_complete = true;
                } else if (!r0_complete) {
                    r0 = vid;
                    r0_complete = true;
                }
            } else if (vid == oneid) {
                if (!l1_complete) {
                    l1 = vid;
                    l1_complete = true;
                } else if (!r1_complete) {
                    r1 = vid;
                    r1_complete = true;
                }
            }
        }

        // TODO: RM
        assert(l0 == std::numeric_limits<unsigned>::max() ||
                l1 == std::numeric_limits<unsigned>::max() ||
                r0 == std::numeric_limits<unsigned>::max() ||
                r1 == std::numeric_limits<unsigned>::max());
        // TODO: End RM

        auto zero_child_ids = ider->get_split_ids(zeroid);
        auto one_child_ids = ider->get_split_ids(oneid);

        // Add parent with two children
        // TODO: memoize these
#if 1
        hcltrs[zeroid] = base::h_clusters::create(2, ncol);
        hcltrs[zeroid]->set_mean(get_thd_data(l0), 0);
        hcltrs[zeroid]->set_mean(get_thd_data(l1), 1);
        hcltrs[zeroid]->set_zeroid(zero_child_ids.first);
        hcltrs[zeroid]->set_oneid(zero_child_ids.second);

        hcltrs[oneid] = base::h_clusters::create(2, ncol);
        hcltrs[oneid]->set_mean(get_thd_data(r0), 0);
        hcltrs[oneid]->set_mean(get_thd_data(r1), 1);
        hcltrs[oneid]->set_zeroid(one_child_ids.first);
        hcltrs[oneid]->set_oneid(one_child_ids.second);
#else
        hcltrs->insert({zeroid, base::h_clusters::create(2, ncol)});
        hcltrs->at(zeroid)->set_mean(get_thd_data(l0), 0);
        hcltrs->at(zeroid)->set_mean(get_thd_data(l1), 1);
        hcltrs->at(zeroid)->set_zeroid(zero_child_ids.first);
        hcltrs->at(zeroid)->set_oneid(zero_child_ids.second);

        hcltrs->insert({oneid, base::h_clusters::create(2, ncol)});
        hcltrs->at(oneid)->set_mean(get_thd_data(r0), 0);
        hcltrs->at(oneid)->set_mean(get_thd_data(r1), 1);
        hcltrs->at(oneid)->set_zeroid(one_child_ids.first);
        hcltrs->at(oneid)->set_oneid(one_child_ids.second);
#endif
    }

    // Now update with new clusters added and delete parent
    for (auto id : remove_cache) {
        hcltrs.erase(id); // Delete
        //hcltrs->erase(id); // Delete
    }

    curr_nclust = hcltrs.size();
    //curr_nclust = hcltrs->size();
}

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
    //hcltrs->at(0)->print_means();

    // Run loop
    bool converged = false;
    size_t iter = 0;
    curr_nclust = 1;

    while (curr_nclust < k) { /*TODO: Or all clusters are inactive*/
        if (curr_nclust > 1)
            init_splits(); // Initialize possible splits

        for (iter = 0; iter < max_iters; iter++) {
            //wake4run(H_SPLIT);
            //wait4complete();

            // Now pick between the cluster splits
            wake4run(H_EM);
            wait4complete();

#if 1
            // TODO: RM -- Testing only
            // Accumulate the counts
            printf("Cluster states: \n");
            print_active_clusters();

            printf("Printing cluster assignment after split:\n");
            base::print_vector(cluster_assignments, nrow);
            accumulate_cluster_counts();
            printf("Printing cluster assignment counts after split:\n");
            base::print(cluster_assignment_counts);
            exit(911);
#endif

            // TODO: Per cluster tolerance early termination
            //if (num_changed == 0 ||
            //((num_changed/(double)nrow)) <= tolerance) {
            //converged = true;
            //break;
            //}
            iter++;
        }
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

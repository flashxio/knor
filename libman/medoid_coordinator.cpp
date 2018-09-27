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

#include "medoid_coordinator.hpp"
#include "medoid_thread.hpp"
#include "io.hpp"
#include "clusters.hpp"
#include "dist_matrix.hpp"
#include "exception.hpp"

namespace knor {
medoid_coordinator::medoid_coordinator(const std::string fn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const kbase::init_t it,
        const double tolerance, const kbase::dist_t dt) :
    coordinator(fn, nrow, ncol, k, max_iters,
            nnodes, nthreads, centers, it, tolerance, dt) {

        cltrs = kbase::clusters::create(k, ncol);
        if (centers) {
            if (kbase::init_t::NONE)
                cltrs->set_mean(centers);
            else {
#ifndef BIND
                printf("[WARNING]: Both init centers "
                        "provided & non-NONE init method specified\n");
#endif
            }
        }

        // Create the pairwise distance matrix
        pw_dm = prune::dist_matrix::create(nrow);
        medoid_energy.assign(k, 0);
        medoids_changed = true; // Run at least 1 iter
        build_thread_state();
    }

void medoid_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;
    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(medoid_thread::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, cltrs, &cluster_assignments[0],
                    fn, pw_dm, &medoid_energy[0]));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
    }
}

std::pair<unsigned, unsigned>
medoid_coordinator::get_rid_len_tup(const unsigned thd_id) {
    unsigned rows_per_thread = nrow / nthreads;
    unsigned start_rid = (thd_id*rows_per_thread);

    if (thd_id == nthreads - 1)
        rows_per_thread += nrow % nthreads;
    return std::pair<unsigned, unsigned>(start_rid, rows_per_thread);
}

void medoid_coordinator::destroy_threads() {
    wake4run(EXIT);
}

// <Thread, within-thread-row-id>
const double* medoid_coordinator::get_thd_data(const unsigned row_id) const {
    unsigned parent_thd = std::upper_bound(thd_max_row_idx.begin(),
            thd_max_row_idx.end(), row_id) - thd_max_row_idx.begin();
    unsigned rows_per_thread = nrow/nthreads; // All but the last thread

    return &((threads[parent_thd]->get_local_data())
            [(row_id-(parent_thd*rows_per_thread))*ncol]);
}

void medoid_coordinator::sanity_check() {
    unsigned chk_nmemb = 0;
    for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) {
        chk_nmemb += cluster_assignment_counts[clust_idx];
    }
    assert(chk_nmemb == nrow);
}

void medoid_coordinator::choose_global_medoids(double* gdata) {
    // Do reduction on these
    std::vector<unsigned> agg_candidate_medoids;
    std::vector<double> agg_candidate_medoid_energy;

    agg_candidate_medoids.assign(cltrs->get_nclust(), -1);
    agg_candidate_medoid_energy.assign(cltrs->get_nclust(),
            std::numeric_limits<double>::max());

    // Accumulate the possible candidates and their energy
    for (auto th : threads) {
        auto t = std::static_pointer_cast<medoid_thread>(th);
        auto cm = t->get_candidate_medoids();
        auto ce = t->get_candidate_energy();

        for (unsigned cid = 0; cid < k; cid++) {
            if (ce[cid] < agg_candidate_medoid_energy[cid]) {
                agg_candidate_medoid_energy[cid] = ce[cid];
                agg_candidate_medoids[cid] = cm[cid];
            }
        }
    }

    // Determine if we need to update
    for (unsigned cid = 0; cid < k; cid++) {
        if (agg_candidate_medoid_energy[cid] < medoid_energy[cid]) {
            medoids_changed = true;
            // Update energy
            medoid_energy[cid] = agg_candidate_medoid_energy[cid];
            // Update new medoid id
            cltrs->get_num_members_v()[cid] = agg_candidate_medoids[cid];
            // Update new (centroid) medoid
            cltrs->set_mean(&(gdata[agg_candidate_medoids[cid]]), cid);
        }
    }
}

void medoid_coordinator::compute_globals() {
    // Always reset here since there's no pruning
    num_changed = 0;
    std::fill(cluster_assignment_counts.begin(),
            cluster_assignment_counts.end(), 0);
    std::fill(medoid_energy.begin(), medoid_energy.end(), 0);

    for (auto th : threads) {
        auto t = std::static_pointer_cast<medoid_thread>(th);

        num_changed += t->get_num_changed();

        for (size_t clust_idx = 0; clust_idx < k; clust_idx++) {
            // Reduction on membership count
            cluster_assignment_counts[clust_idx] +=
                t->get_local_clusters()->get_num_members(clust_idx);

            // Reduction on medoid energy
            medoid_energy[clust_idx] += t->get_local_medoid_energy()[clust_idx];
        }
    }

    assert(num_changed <= nrow);
}

double medoid_coordinator::reduction_on_cuml_sum() {
    double tot = 0;
    for (thread_iter it = threads.begin(); it != threads.end(); ++it)
        tot += (*it)->get_cuml_dist();
    return tot;
}

void medoid_coordinator::wake4run(const thread_state_t state) {
    pending_threads = nthreads;
    for (unsigned thd_id = 0; thd_id < threads.size(); thd_id++)
        threads[thd_id]->wake(state);
}

void medoid_coordinator::set_thread_clust_idx(const unsigned clust_idx) {
    for (thread_iter it = threads.begin(); it != threads.end(); ++it)
        (*it)->set_clust_idx(clust_idx);
}

// Default
void medoid_coordinator::forgy_init() {
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, nrow-1);

    for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) { // 0...k
        unsigned rand_idx = distribution(generator);
        cltrs->set_mean(get_thd_data(rand_idx), clust_idx);

        // NOTE: Use the member count as the ID of the chosen
        cltrs->get_num_members_v()[clust_idx] = rand_idx;
    }
}

void medoid_coordinator::run_init() {
    if (_init_t == kbase::init_t::FORGY) {
        forgy_init();
        // Run one EM step to assign samples to a cluster
        wake4run(EM);
        wait4complete();
        compute_globals();
    } else {
        throw kbase::parameter_exception("Unsupported initialization type");
    }
}

/**
 * Main driver
 */
kbase::kmeans_t medoid_coordinator::run(
        double* allocd_data, const bool numa_opt) {
#ifdef PROFILER
    ProfilerStart("libman/medoid_coordinator.perf");
#endif

    if (numa_opt)
        throw knor::base::not_implemented_exception();

    set_thread_data_ptr(allocd_data); // Offset taken for each thread
    pw_dm->compute_pairwise_dist(allocd_data, ncol, _dist_t);

    struct timeval start, end;
    gettimeofday(&start , NULL);
    run_init(); // Initialize clusters

#ifndef BIND
    std::cout << "AFTER INIT\n";
    std::cout << "Means are: \n";
    cltrs->print_means();
    std::cout << "Cluster counts:\n";
    kbase::print_vector(cluster_assignment_counts);
    std::cout << "Medoid Energy:\n";
    kbase::print_vector(medoid_energy);
    sanity_check();
    std::cout << "END INIT\n\n";
#endif

    // Run kmeans loop
    bool converged = false;
    size_t iter = 0;

    if (max_iters > 0)
        iter++;

    while (iter <= max_iters && max_iters > 0) {
        wake4run(MEDOID);
        wait4complete();
        // (Possibly) Sets: 1. new medoids, new energy
        choose_global_medoids(allocd_data);

#ifndef BIND
        std::cout << "Medoid Energy:\n";
        kbase::print_vector(medoid_energy);
        std::cout << "Medoids are: \n";
        cltrs->print_means();
#endif

        if (medoids_changed) {
            // Run kmeans step
            wake4run(EM);
            wait4complete();
            compute_globals();
#ifndef BIND
            sanity_check();
#endif

            medoids_changed = false; // Reset

#ifndef BIND
            printf("Cluster assignment counts: \n");
            kbase::print_vector(cluster_assignment_counts);
            printf("\n******************************************\n");
#endif
        } else {
            converged = true;
            break;
        }

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
        printf("K-medoids converged in %lu iterations\n", iter);
#endif
    } else {
#ifndef BIND
        printf("[Warning]: K-medoids failed to converge in %lu"
            " iterations\n", iter);
#endif
    }

#ifndef BIND
    printf("Final cluster counts: \n");
    kbase::print_vector(cluster_assignment_counts);
    printf("\n******************************************\n");
#endif

    return kbase::kmeans_t(this->nrow, this->ncol, iter, this->k,
            &cluster_assignments[0], &cluster_assignment_counts[0],
            cltrs->get_means());
}

medoid_coordinator::~medoid_coordinator() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it)
        (*it)->destroy_numa_mem();

    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&mutex);
    pthread_mutexattr_destroy(&mutex_attr);
    destroy_threads();
}

void const medoid_coordinator::print_thread_data() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it) {
#ifndef BIND
        std::cout << "\nThd: " << (*it)->get_thd_id() << std::endl;
#endif
        (*it)->print_local_data();
    }
}

// Testing
void const medoid_coordinator::print_thread_start_rids() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it) {
#ifndef BIND
        printf("\nThd: %u, start_rid: %lu\n", (*it)->get_thd_id(),
            (*it)->get_start_rid());
#endif
    }
}
}

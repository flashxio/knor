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
#include "medoid.hpp"
#include "io.hpp"
#include "clusters.hpp"
#include "exception.hpp"

namespace knor {
medoid_coordinator::medoid_coordinator(const std::string fn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const kbase::init_t it,
        const double tolerance, const kbase::dist_t dt,
        const double sample_rate) :
    coordinator(fn, nrow, ncol, k, max_iters,
            nnodes, nthreads, centers, it, tolerance, dt) {

        cltrs = kbase::clusters::create(k, ncol);
        if (centers) {
            if (it == kbase::init_t::NONE)
                cltrs->set_mean(centers);
            else {
#ifndef BIND
                printf("[WARNING]: Both init centers "
                        "provided & non-NONE init method specified\n");
#endif
            }
        }

        // Create the pairwise distance matrix
        membership.resize(k);
        medoid_energy.assign(k, 0);
        medoids_changed = true; // Run at least 1 iter

        // At least 1 element should be processed by each threads!
        this->sample_rate = std::max(.2, sample_rate);
        build_thread_state();
    }

void medoid_coordinator::populate_membership() {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned cid = 0; cid < k; cid++) {
        for (size_t rid = 0; rid < nrow; rid++) {
            if (cluster_assignments[rid] == cid)
                membership[cid].push_back(rid);
        }
    }
}

void medoid_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;
    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(medoid::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, cltrs, &cluster_assignments[0],
                    fn, sample_rate));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
        std::static_pointer_cast<medoid>(threads[thd_id])->set_coordinator(this);
    }
}

void medoid_coordinator::sanity_check() {
    unsigned chk_nmemb = 0;
    for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) {
        chk_nmemb += cluster_assignment_counts[clust_idx];
    }
    assert(chk_nmemb == nrow);
}

// TODO: Maybe use an omp-style vector reduction
void medoid_coordinator::choose_global_medoids(double* gdata) {
    // Do reduction on these
    std::vector<unsigned> agg_candidate_medoids;
    std::vector<double> agg_candidate_medoid_energy;

    agg_candidate_medoids.assign(cltrs->get_nclust(), -1);
    agg_candidate_medoid_energy.assign(cltrs->get_nclust(),
            std::numeric_limits<double>::max());

    // Accumulate the possible candidates and their energy
    for (auto const& th : threads) {
        auto t = std::static_pointer_cast<medoid>(th);
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

            if (NULL == gdata)
                cltrs->set_mean(get_thd_data(agg_candidate_medoids[cid]), cid);
            else
                cltrs->set_mean(&(gdata[agg_candidate_medoids[cid]*ncol]), cid);
        }
    }
}

void medoid_coordinator::compute_globals() {
    // Compute the membership assignments
    populate_membership();
    // Always reset here since there's no pruning
    num_changed = 0;
    std::fill(cluster_assignment_counts.begin(),
            cluster_assignment_counts.end(), 0);
    std::fill(medoid_energy.begin(), medoid_energy.end(), 0);

    for (auto const& th : threads) {
        auto t = std::static_pointer_cast<medoid>(th);

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
kbase::cluster_t medoid_coordinator::run(
        double* allocd_data, const bool numa_opt) {
#ifdef PROFILER
    ProfilerStart("libman/medoid_coordinator.perf");
#endif

    if (numa_opt)
        throw kbase::not_implemented_exception();

    if (!numa_opt && NULL == allocd_data) {
        wake4run(ALLOC_DATA);
        wait4complete();
    } else if (allocd_data) { // No NUMA opt
        set_thread_data_ptr(allocd_data); // Offset taken for each thread
    } // Do nothing for numa_opt .. done in binding/knori.hpp

    struct timeval start, end;
    gettimeofday(&start , NULL);
    run_init(); // Initialize clusters

    // Run kmeans loop
    bool converged = false;
    size_t iter = 0;

    if (max_iters > 0)
        iter++;

    while (iter <= max_iters && max_iters > 0) {
#ifndef BIND
        printf("Medoid step ...\n");
#endif
        wake4run(MEDOID);
        wait4complete();
        // (Possibly) Sets: 1. new medoids, new energy
#ifndef BIND
        printf("Choosing global medoids ...\n");
#endif
        choose_global_medoids(allocd_data);

#ifndef BIND
        printf("Medoid Energy:\n");
        kbase::print(medoid_energy);
#endif

        if (medoids_changed) {
            // Run kmeans step
#ifndef BIND
            printf("EM step ...\n");
#endif
            clear_membership();
            wake4run(EM);
            wait4complete();
            compute_globals();

#ifndef BIND
            printf("Cluster assignment counts: \n");
            kbase::print(cluster_assignment_counts);
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
    kbase::print(cluster_assignment_counts);
    printf("\n******************************************\n");
#endif

    return kbase::cluster_t(this->nrow, this->ncol, iter, this->k,
            &cluster_assignments[0], &cluster_assignment_counts[0],
            cltrs->get_means());
}
}

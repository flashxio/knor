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

#include "fcm_coordinator.hpp"
#include "fcm.hpp"
#include "io.hpp"
#include "clusters.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace knor {
fcm_coordinator::fcm_coordinator(const std::string fn, const size_t nrow,
        const size_t ncol, const unsigned k, const unsigned max_iters,
        const unsigned nnodes, const unsigned nthreads,
        const double* centers, const kbase::init_t it,
        const double tolerance, const kbase::dist_t dt,
        const unsigned fuzzindex) :
    coordinator(fn, nrow, ncol, k, max_iters,
            nnodes, nthreads, centers, it, tolerance, dt),
            fuzzindex(fuzzindex) {

#ifdef _OPENMP
        omp_set_num_threads(nthreads);
#endif

        this->centers = base::dense_matrix<double>::create(k, ncol);
        this->prev_centers = base::dense_matrix<double>::create(k, ncol);
        this->um = base::dense_matrix<double>::create(k, nrow);

        if (centers)
            this->centers->set(centers);

        build_thread_state();
    }

void fcm_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;
    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(
                fcm::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second,
                    ncol, k, fuzzindex, um, centers, fn, _dist_t));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
    }
}

fcm_coordinator::~fcm_coordinator() {
    delete (centers);
    delete (prev_centers);
    delete (um);
}

void fcm_coordinator::forgy_init() {
#if 1
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, nrow-1);

    for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) { // 0...k
        unsigned rand_idx = distribution(generator);
        centers->set_row(get_thd_data(rand_idx), clust_idx);
    }
#else
    // Testing for iris with k = 3
    assert(k == 3);
    centers->set_row(get_thd_data(91),0);
    centers->set_row(get_thd_data(63),1);
    centers->set_row(get_thd_data(103),2);
#endif
}

void fcm_coordinator::update_contribution_matrix() {
    std::vector<double> colsums;
    um->sum(0, colsums); // k x nrow
    um->div_eq_pow(colsums, 0, fuzzindex);
}

void fcm_coordinator::update_centers() {
    if (threads.size() == 1) {
        centers->copy_from(std::static_pointer_cast<fcm>(
                    threads[0])->get_innerprod());
    } else {
        auto sum = *(std::static_pointer_cast<fcm>(threads[0])->get_innerprod()) +
                *(std::static_pointer_cast<fcm>(threads[1])->get_innerprod());
        centers->copy_from(sum);
        delete (sum);

        for (size_t tid = 2; tid < threads.size(); tid++) {
            *centers += *(std::static_pointer_cast<fcm>(
                            threads[tid])->get_innerprod());
        }
    }

    // Take sum along axis
    std::vector<double> sum;
    um->sum(1, sum);
    centers->div_eq(sum, 1);
}

/**
 * Main driver
 */
base::cluster_t fcm_coordinator::run(double* allocd_data,
        const bool numa_opt) {
#ifdef PROFILER
    ProfilerStart("fcm_coordinator.perf");
#endif

    if (!allocd_data) {
        wake4run(ALLOC_DATA);
        wait4complete();
    } else if (allocd_data) {
        set_thread_data_ptr(allocd_data);
    }

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
        std::cout << "Running iteration: "  << iter << std::endl;
#endif
        // Compute new um
        wake4run(E);
        wait4complete();

        update_contribution_matrix();

#if VERBOSE
#ifndef BIND
        std::cout << "After Estep contribution matrix is: \n";
        um->print();
#endif
#endif
        // Compute new centers
        prev_centers->copy_from(centers);

        wake4run(M);
        wait4complete();
        update_centers();

#if VERBOSE
#ifndef BIND
        std::cout << "After Mstep centers are: \n";
        centers->print();
#endif
#endif

        auto diff = (*centers - *prev_centers);
        auto frob_norm = diff->frobenius_norm();
#ifndef BIND
        std::cout << "Centers frob diff: " << frob_norm << "\n\n";
#endif

        if (frob_norm < tolerance) {
            converged = true;
            delete (diff);
            break;
        }

        delete (diff);
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
        printf("Fuzzy C-means converged in %lu iterations\n", iter);
#endif
    } else {
#ifndef BIND
        printf("[Warning]: Fuzzy C-means failed to converge in %lu"
            " iterations\n", iter);
#endif
    }

    um->argmax(1, cluster_assignments);
    cluster_assignment_counts.assign(k, 0);
    for (auto const& i : cluster_assignments)
        cluster_assignment_counts[i]++;

#ifndef BIND
    printf("Final cluster assignment count:\n");
    kbase::print(cluster_assignment_counts);
    printf("\n******************************************\n");
#endif

    return kbase::cluster_t(this->nrow, this->ncol, iter, this->k,
            &cluster_assignments[0], &cluster_assignment_counts[0],
            centers->as_vector());
}
}

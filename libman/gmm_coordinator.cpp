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
#include <algorithm>

#include "gmm_coordinator.hpp"
#include "gmm.hpp"
#include "dense_matrix.hpp"
#include "io.hpp"

namespace knor {
gmm_coordinator::gmm_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k,
                const unsigned max_iters, double* mu_k,
                const unsigned nnodes, const unsigned nthreads,
                const base::init_t it,
                const double tolerance, const base::dist_t dt,
                const double cov_regularizer) :
    coordinator(fn, nrow, ncol, k, max_iters,
            nnodes, nthreads, mu_k, it, tolerance, dt) {

        this->cov_regularizer = cov_regularizer;
        this->k = k;
        this->mu_k = base::dense_matrix<double>::create(k, ncol);

        for (unsigned i = 0; i < k; i++)
            sigma_k.push_back(base::dense_matrix<double>::create(ncol, ncol));

        this->P_nk = base::dense_matrix<double>::create(nrow, k);
        this->Pk.resize(k);

        if (mu_k) {
            this->_init_t = base::init_t::NONE;
            this->mu_k->set(mu_k);
        }

        build_thread_state();
    }

void gmm_coordinator::build_thread_state() {
    // NUMA node affinity binding policy is round-robin
    unsigned thds_row = nrow / nthreads;
    for (unsigned thd_id = 0; thd_id < nthreads; thd_id++) {
        std::pair<unsigned, unsigned> tup = get_rid_len_tup(thd_id);
        thd_max_row_idx.push_back((thd_id*thds_row) + tup.second);
        threads.push_back(gmm::create((thd_id % nnodes),
                    thd_id, tup.first, tup.second, ncol, fn, _dist_t));
        threads[thd_id]->set_parent_cond(&cond);
        threads[thd_id]->set_parent_pending_threads(&pending_threads);
        threads[thd_id]->start(WAIT); // Thread puts itself to sleep
        std::static_pointer_cast<gmm>(threads[thd_id])->set_alg_metadata(
                k, mu_k, &(sigma_k[0]), P_nk, &Pk[0]);
    }
}

std::pair<unsigned, unsigned>
gmm_coordinator::get_rid_len_tup(const unsigned thd_id) {
    unsigned rows_per_thread = nrow / nthreads;
    unsigned start_rid = (thd_id*rows_per_thread);

    if (thd_id == nthreads - 1)
        rows_per_thread += nrow % nthreads;
    return std::pair<unsigned, unsigned>(start_rid, rows_per_thread);
}

void gmm_coordinator::destroy_threads() {
    wake4run(EXIT);
}

// <Thread, within-thread-row-id>
const double* gmm_coordinator::get_thd_data(const unsigned row_id) const {
    // TODO: Cheapen
    unsigned parent_thd = std::upper_bound(thd_max_row_idx.begin(),
            thd_max_row_idx.end(), row_id) - thd_max_row_idx.begin();
    unsigned rows_per_thread = nrow/nthreads; // All but the last thread

    return &((threads[parent_thd]->get_local_data())
            [(row_id-(parent_thd*rows_per_thread))*ncol]);
}

void gmm_coordinator::update_clusters() {
    //num_changed = 0; // Always reset here since there's no pruning
    //cltrs->clear();

    //// Serial aggreate of OMP_MAX_THREADS vectors
    //for (thread_iter it = threads.begin(); it != threads.end(); ++it) {
        //// Updated the changed cluster count
        //num_changed += (*it)->get_num_changed();
        //// Summation for cluster

        //cltrs->peq((*it)->get_local_clusters());
    //}

    //unsigned chk_nmemb = 0;
    //for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) {
        //cltrs->finalize(clust_idx);
        //cluster_assignment_counts[clust_idx] =
            //cltrs->get_num_members(clust_idx);
        //chk_nmemb += cluster_assignment_counts[clust_idx];
    //}

    //assert(chk_nmemb == nrow);
    //assert(num_changed <= nrow);
}

double gmm_coordinator::reduction_on_cuml_sum() {
    double tot = 0;
    for (thread_iter it = threads.begin(); it != threads.end(); ++it)
        tot += (*it)->get_cuml_dist();
    return tot;
}

void gmm_coordinator::wake4run(const thread_state_t state) {
    pending_threads = nthreads;
    for (unsigned thd_id = 0; thd_id < threads.size(); thd_id++)
        threads[thd_id]->wake(state);
}

void gmm_coordinator::set_thread_clust_idx(const unsigned clust_idx) {
    for (thread_iter it = threads.begin(); it != threads.end(); ++it)
        (*it)->set_clust_idx(clust_idx);
}

void gmm_coordinator::set_thd_dist_v_ptr(double* v) {
    for (unsigned thd_id = 0; thd_id < threads.size(); thd_id++) {
        pthread_mutex_lock(&mutex);
        threads[thd_id]->set_dist_v_ptr(v);
        pthread_mutex_unlock(&mutex);
    }
}

void gmm_coordinator::kmeanspp_init() {
    struct timeval start, end;
    gettimeofday(&start , NULL);

    std::vector<double> dist_v;
    dist_v.assign(nrow, std::numeric_limits<double>::max());
    set_thd_dist_v_ptr(&dist_v[0]);

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, nrow-1);

    // Choose c1 uniformly at random
    unsigned selected_idx = distribution(generator);
    mu_k->set_row(get_thd_data(selected_idx), 0);
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
        if (++clust_idx >= k)  // No more  needed
            break;

        for (size_t row = 0; row < nrow; row++) {
            cuml_dist -= dist_v[row];
            if (cuml_dist <= 0) {
                mu_k->set_row(get_thd_data(row), clust_idx);
                cluster_assignments[row] = clust_idx;
                break;
            }
        }
        assert(cuml_dist <= 0);
    }

    gettimeofday(&end, NULL);
}

void gmm_coordinator::forgy_init() {
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, nrow-1);

    for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) { // 0...k
        unsigned rand_idx = distribution(generator);
        mu_k->set_row(get_thd_data(rand_idx), clust_idx);
    }
}

/**
  * A random probabilistic fill such where each row sums to 1 as in a pdf
**/
void gmm_coordinator::random_prob_fill(base::dense_matrix<double>* dm,
        const double mix, const double max) {

    std::uniform_real_distribution<double> distribution(mix, max);
    std::default_random_engine generator;
    const size_t nrow = dm->get_nrow();
    const size_t ncol = dm->get_ncol();

    for (size_t row = 0; row < nrow; row++) {
        double sum = 0;
        for (size_t col = 0; col < ncol; col++) {
            double val = distribution(generator);
            sum += val;
            dm->as_vector()[row*ncol+col] = val;
        }
        // Normalize row
        for (size_t col = 0; col < dm->get_ncol(); col++)
            dm->as_vector()[row*ncol+col] /= sum;
    }
}

void gmm_coordinator::random_prob_fill(std::vector<double>& v,
        const double min, const double max) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(min, max);

    double sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        double val = distribution(generator);
        sum += val;
        v[i] = val;
    }

    for (size_t i = 0; i < v.size(); i++)
        v[i] /= sum;
}

void gmm_coordinator::random_init() {
    forgy_init();

    random_prob_fill(Pk);
    random_prob_fill(P_nk);
    for(auto cov : sigma_k)
        random_prob_fill(cov);
}

void gmm_coordinator::run_init() {
    switch(_init_t) {
        case kbase::init_t::RANDOM:
            random_init();
            break;
        case base::init_t::FORGY:
            forgy_init();
            break;
        case base::init_t::PLUSPLUS:
            kmeanspp_init();
            break;
        case base::init_t::NONE:
            break;
        default:
            throw std::runtime_error("Unknown initialization type");
    }
}

/**
 * Main driver for kmeans
 */
base::gmm_t gmm_coordinator::soft_run(double* allocd_data) {
#ifdef PROFILER
    ProfilerStart("libman/gmm_coordinator.perf");
#endif

    if (NULL == allocd_data) {
        wake4run(ALLOC_DATA);
        wait4complete();
    } else if (allocd_data) {
        set_thread_data_ptr(allocd_data);
    }

    struct timeval start, end;
    gettimeofday(&start , NULL);
    run_init(); // Initialize clusters

    ////////////////////////////////////////////////////////////////////////////
    std::cout << "\nAFTER INIT:\nPk:\n";
    base::print_vector(Pk);
    std::cout << "mu_k:\n"; mu_k->print();
    std::cout << "P_nk:\n"; P_nk->print();
    std::cout << "cov_regularizer: " << cov_regularizer << "\n"
        << "sigma_k:\n";
    for (auto cov : sigma_k) {
        cov->print();
        std::cout << "\n";
    }
    std::cout << "\n\n";
    std::cout << "DATA:\n";
    print_thread_data();
    exit(911);
    ////////////////////////////////////////////////////////////////////////////

    // Run kmeans loop
    bool converged = false;
    size_t iter = 0;

    if (max_iters > 0)
        iter++;

    while (iter <= max_iters && max_iters > 0) {
        if (iter == 1)
            clear_cluster_assignments();

        wake4run(EM);
        wait4complete();

        update_clusters();

#if VERBOSE
#ifndef BIND
        printf("Cluster assignment counts: \n");
#endif
        base::print_vector(cluster_assignment_counts);
#endif

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
        base::time_diff(start, end));
    printf("\n******************************************\n");
#endif
    if (converged) {
#ifndef BIND
        printf("K-means converged in %lu iterations\n", iter);
#endif
    } else {
#ifndef BIND
        printf("[Warning]: K-means failed to converge in %lu"
            " iterations\n", iter);
#endif
    }

#ifndef BIND
    printf("Final cluster counts: \n");
    base::print_vector(cluster_assignment_counts);
    printf("\n******************************************\n");
#endif

    return base::gmm_t(this->nrow, this->ncol, iter, this->k,
            mu_k->as_pointer(), this->sigma_k,
            P_nk->as_pointer(), &Pk[0]);
}

gmm_coordinator::~gmm_coordinator() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it)
        (*it)->destroy_numa_mem();

    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&mutex);
    pthread_mutexattr_destroy(&mutex_attr);
    destroy_threads();

    // destroy metadata
    delete mu_k; // estimated guassians (k means)
    for (size_t i = 0; i < sigma_k.size(); i++)
        delete sigma_k[i];
    delete P_nk; // responsibility matrix (nxk)
}

void const gmm_coordinator::print_thread_data() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it) {
#ifndef BIND
        std::cout << "\nThd: " << (*it)->get_thd_id() << std::endl;
#endif
        (*it)->print_local_data();
    }
}

// Testing
void const gmm_coordinator::print_thread_start_rids() {
    thread_iter it = threads.begin();
    for (; it != threads.end(); ++it) {
#ifndef BIND
        printf("\nThd: %u, start_rid: %lu\n", (*it)->get_thd_id(),
            (*it)->get_start_rid());
#endif
    }
}
}

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
#include "linalg.hpp"

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

        for (unsigned i = 0; i < k; i++) {
            sigma_k.push_back(base::dense_matrix<double>::create(ncol, ncol));
            inv_sigma_k.push_back(
                    base::dense_matrix<double>::create(ncol, ncol));
        }

        this->P_nk = base::dense_matrix<double>::create(nrow, k);
        this->Pk.resize(k);
        this->dets.resize(k);
        this->Px.assign(nrow, 0);
        Pnk_sum.resize(k);

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
                k, mu_k, &(sigma_k[0]), P_nk, &Pk[0],
                &inv_sigma_k[0], &dets[0], &Px[0]);
    }
}

void gmm_coordinator::update_clusters() {
    L = 0; // Reset
    Pnk_sum.assign(k, 0);
    for (size_t row = 0; row < nrow; row++) {
        for (size_t col = 0; col < ncol; col++) {
            Pnk_sum[col] += P_nk->get(row, col);
        }
    }

    for (auto const& th : threads) {
        // Now L is complete
        L += std::static_pointer_cast<gmm>(th)->get_L();
    }


    // Updates pointer data in threads
    compute_shared_linalg();
    Px.assign(nrow, 0); // reset
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

void gmm_coordinator::compute_cov_mat() {
    throw base::not_implemented_exception(); // TODO
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < mu_k->get_nrow(); i++) {
    }
}

void gmm_coordinator::random_partition_init() {
    forgy_init();

    random_prob_fill(Pk);
    random_prob_fill(P_nk);
    for (auto const& cov : sigma_k)
        random_prob_fill(cov);
}

void gmm_coordinator::compute_shared_linalg() {
    // Compute Inverse sigma
    dets.clear();
    for (size_t cid = 0; cid < k; cid++) {
        base::linalg::inverse(sigma_k[cid]->as_pointer(),
                inv_sigma_k[cid]->as_pointer(), sigma_k[cid]->get_nrow());
        dets[cid] = base::linalg::determinant(sigma_k[cid]->as_pointer(),
                sigma_k[cid]->get_nrow(), sigma_k[cid]->get_nrow());
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
#ifndef BIND
    std::cout << "\nAFTER INIT:\nPk:\n";
    base::print(Pk);
    std::cout << "mu_k:\n"; mu_k->print();
    std::cout << "P_nk:\n"; P_nk->print();
    std::cout << "cov_regularizer: " << cov_regularizer << "\n"
        << "sigma_k:\n";
    for (auto const& cov : sigma_k) {
        cov->print();
        std::cout << "\n";
    }
    std::cout << "\n\n";
    std::cout << "DATA:\n";
    print_thread_data();
    exit(911);
#endif
    ////////////////////////////////////////////////////////////////////////////

    // Run kmeans loop
    bool converged = false;
    size_t iter = 0;

    if (max_iters > 0)
        iter++;

    while (iter <= max_iters && max_iters > 0) {
        if (iter == 1)
            clear_cluster_assignments();

        wake4run(E);
        wait4complete();

        update_clusters();
        wake4run(M);

#if VERBOSE
#ifndef BIND
        printf("Cluster assignment counts: \n");
#endif
        base::print(cluster_assignment_counts);
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
        printf("GMM converged in %lu iterations\n", iter);
#endif
    } else {
#ifndef BIND
        printf("[Warning]: GMM failed to converge in %lu"
            " iterations\n", iter);
#endif
    }

#ifndef BIND
    printf("Final cluster counts: \n");
    base::print(cluster_assignment_counts);
    printf("\n******************************************\n");
#endif

    return base::gmm_t(this->nrow, this->ncol, iter, this->k,
            mu_k->as_pointer(), this->sigma_k,
            P_nk->as_pointer(), &Pk[0]);
}

gmm_coordinator::~gmm_coordinator() {
    // destroy metadata
    delete mu_k; // estimated guassians (k means)
    for (size_t i = 0; i < sigma_k.size(); i++) {
        delete sigma_k[i];
        delete inv_sigma_k[i];
    }
    delete P_nk; // responsibility matrix (nxk)
}
}

/*
 * Copyright 2016 neurodata (http://neurodata.io/)
 * Written by Disa Mhembere (disa@jhu.edu)
 *
 * This file is part of k-par-means
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

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

#include <omp.h>

#include "kmeans.hpp"
#include "kcommon.hpp"
#include "exception.hpp"

#define KM_TEST 0
#define VERBOSE 0

namespace kpmprune = kpmeans::prune;
namespace kpmbase = kpmeans::base;

namespace {

static size_t NUM_COLS;
static size_t K;
static size_t NUM_ROWS;
short OMP_MAX_THREADS;
static size_t g_num_changed = 0;
static struct timeval start, end;
static kpmbase::init_type_t g_init_type;
static kpmbase::dist_type_t g_dist_type;

/**
 * \brief This initializes clusters by randomly choosing sample
 *		membership in a cluster.
 * See: http://en.wikipedia.org/wiki/K-means_clustering#Initialization_methods
 *	\param cluster_assignments Which cluster each sample falls into.
 */
void random_partition_init(unsigned* cluster_assignments,
        const double* matrix,
        std::shared_ptr<kpmbase::clusters> clusters,
        const size_t num_rows,
        const size_t num_cols, const unsigned k) {
    BOOST_LOG_TRIVIAL(info) << "Random init start";

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, k-1);

//#pragma omp parallel for shared(cluster_assignments)
    for (size_t row = 0; row < num_rows; row++) {
        unsigned asgnd_clust = distribution(generator);

        clusters->add_member(&matrix[row*num_cols], asgnd_clust);
        cluster_assignments[row] = asgnd_clust;
    }

    // NOTE: M-Step called in compute func to update cluster counts & centers
#if VERBOSE
#ifndef BIND
    printf("After rand paritions cluster_asgns: ");
#endif
    print_arr(cluster_assignments, num_rows);
#endif
    BOOST_LOG_TRIVIAL(info) << "Random init end\n";
}

/**
 * \brief Forgy init takes `K` random samples from the matrix
 *		and uses them as cluster centers.
 * \param matrix the flattened matrix who's rows are being clustered.
 * \param clusters The cluster centers (means) flattened matrix.
 */
void forgy_init(const double* matrix,
        std::shared_ptr<kpmbase::clusters> clusters,
        const size_t num_rows, const size_t num_cols, const unsigned k) {

    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(0, num_rows-1);

    BOOST_LOG_TRIVIAL(info) << "Forgy init start";

    for (unsigned clust_idx = 0; clust_idx < k; clust_idx++) { // 0...K
        size_t rand_idx = distribution(generator);
        clusters->set_mean(&matrix[rand_idx*num_cols], clust_idx);
    }

    BOOST_LOG_TRIVIAL(info) << "Forgy init end";
}

#if KM_TEST
std::string s (const double d) {
    if (d == std::numeric_limits<double>::max())
        return "max";
    else
        return std::to_string(d);
}
#endif

/**
 * \brief A parallel version of the kmeans++ initialization alg.
 *  See: http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf for algorithm
 */
static void kmeanspp_init(const double* matrix,
        kpmbase::prune_clusters::ptr clusters,
        unsigned* cluster_assignments) {

    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(0, NUM_ROWS-1);

    // Choose c1 uniformly at random
    size_t selected_idx = distribution(generator);
    std::vector<double> dist_v;
    dist_v.assign(NUM_ROWS, std::numeric_limits<double>::max());

    clusters->set_mean(&matrix[selected_idx*NUM_COLS], 0);
    dist_v[selected_idx] = 0.0;
    cluster_assignments[selected_idx] = 0;

#if KM_TEST
    BOOST_LOG_TRIVIAL(info) << "\nChoosing "
        << selected_idx << " as center K = 0";
#endif

    unsigned clust_idx = 0; // The number of clusters assigned

    std::uniform_real_distribution<double> ur_distribution(0.0, 1.0);

    // Choose next center c_i with weighted prob
    while (true) {
        double cum_dist = 0;
#pragma omp parallel for reduction(+:cum_dist) shared (dist_v, cluster_assignments)
        for (size_t row = 0; row < NUM_ROWS; row++) {
            // Prune in kms++ possible using
            double dist = dist_comp_raw(&matrix[row*NUM_COLS],
                    &((clusters->get_means())[clust_idx*NUM_COLS]),
                    NUM_COLS, g_dist_type);

            if (dist < dist_v[row]) { // Found a closer cluster than before
                dist_v[row] = dist;
                cluster_assignments[row] = clust_idx;
            }
            cum_dist += dist_v[row];
        }

        cum_dist = (cum_dist * ur_distribution(generator)) / (RAND_MAX - 1.0);
        if (++clust_idx >= K)  // No more centers needed
            break;

        for (size_t i = 0; i < NUM_ROWS; i++) {
            cum_dist -= dist_v[i];
            if (cum_dist <= 0) {
#if KM_TEST
                BOOST_LOG_TRIVIAL(info) << "Choosing "
                    << i << " as center K = " << clust_idx;
#endif
                cluster_assignments[i] = clust_idx;
                clusters->set_mean(&(matrix[i*NUM_COLS]), clust_idx);
                break;
            }
        }
        assert (cum_dist <= 0);
    }

#if VERBOSE
    BOOST_LOG_TRIVIAL(info) << "\nCluster centers after kmeans++";
    clusters->print_means();
#endif
}

/**
 * \brief Update the cluster assignments while recomputing distance matrix.
 * \param matrix The flattened matrix who's rows are being clustered.
 * \param clusters The cluster centers (means) flattened matrix.
 *	\param cluster_assignments Which cluster each sample falls into.
 */
static void EM_step(const double* matrix, kpmbase::prune_clusters::ptr cls,
        unsigned* cluster_assignments, size_t* cluster_assignment_counts,
        kpmbase::thd_safe_bool_vector::ptr recalculated_v,
        std::vector<double>& dist_v,
        kpmprune::dist_matrix::ptr dm, const bool prune_init=false) {

    std::vector<kpmbase::clusters::ptr> pt_cl(OMP_MAX_THREADS);
    // Per thread changed cluster count. OMP_MAX_THREADS
    std::vector<size_t> pt_num_change(OMP_MAX_THREADS);

    for (int i = 0; i < OMP_MAX_THREADS; i++)
        pt_cl[i] = kpmbase::clusters::create(K, NUM_COLS);

#pragma omp parallel for firstprivate(matrix, pt_cl)\
    shared(cluster_assignments, recalculated_v, dist_v)
    for (size_t row = 0; row < NUM_ROWS; row++) {
        unsigned old_clust = cluster_assignments[row];
        size_t offset = row*NUM_COLS;

        if (prune_init) {
            double dist = std::numeric_limits<double>::max();

            for (unsigned clust_idx = 0; clust_idx < K; clust_idx++) {
                dist = dist_comp_raw(&matrix[offset],
                        &(cls->get_means()[clust_idx*NUM_COLS]), NUM_COLS,
                        g_dist_type);

                if (dist < dist_v[row]) {
                    dist_v[row] = dist;
                    cluster_assignments[row] = clust_idx;
                }
            }

        } else {
            recalculated_v->set(row, false);
            dist_v[row] += cls->get_prev_dist(cluster_assignments[row]);

            if (dist_v[row] <= cls->get_s_val(cluster_assignments[row])) {
                // Skip all rows
            } else {
                for (unsigned clust_idx = 0; clust_idx < K; clust_idx++) {

                    if (dist_v[row] <= dm->get(cluster_assignments[row],
                                clust_idx)) {
                        // Skip this cluster
                        continue;
                    }

                    if (!recalculated_v->get(row)) {
                        dist_v[row] = dist_comp_raw(&matrix[offset],
                                &(cls->get_means()[cluster_assignments[row]*NUM_COLS]),
                                NUM_COLS, g_dist_type);
                        recalculated_v->set(row, true);
                    }

                    if (dist_v[row] <= dm->get(cluster_assignments[row],
                                clust_idx)) {
                        // Skip this cluster
                        continue;
                    }

                    // Track 5
                    double jdist = dist_comp_raw(&matrix[offset],
                            &(cls->get_means()[clust_idx*NUM_COLS]), NUM_COLS, g_dist_type);

                    if (jdist < dist_v[row]) {
                        dist_v[row] = jdist;
                        cluster_assignments[row] = clust_idx;
                    }
                } // endfor
            }
        }

        BOOST_VERIFY(cluster_assignments[row] >= 0 &&
                cluster_assignments[row] < K);

        if (prune_init) {
            pt_num_change[omp_get_thread_num()]++;
            pt_cl[omp_get_thread_num()]->add_member(&matrix[offset],
                    cluster_assignments[row]);
        } else if (old_clust != cluster_assignments[row]) {
            pt_num_change[omp_get_thread_num()]++;
            pt_cl[omp_get_thread_num()]->swap_membership(&matrix[offset],
                    old_clust, cluster_assignments[row]);
        }
    }

#if VERBOSE
    BOOST_LOG_TRIVIAL(info) << "Clearing/unfinalizing cluster centers ...";
#endif

    if (prune_init) {
        cls->clear();
    } else {
        cls->set_prev_means();
        cls->unfinalize_all();
    }

    // Serial aggreate of OMP_MAX_THREADS vectors
    // TODO: Pool these
    for (int thd = 0; thd < OMP_MAX_THREADS; thd++) {
        // Updated the changed cluster count
        g_num_changed += pt_num_change[thd];
        // Summation for cluster centers
        cls->peq(pt_cl[thd]);
    }

    size_t chk_nmemb = 0;
    for (unsigned clust_idx = 0; clust_idx < K; clust_idx++) {
        cls->finalize(clust_idx);
        cls->set_prev_dist(kpmbase::eucl_dist(
                    &(cls->get_means()[clust_idx*NUM_COLS]),
                    &(cls->get_prev_means()[clust_idx*NUM_COLS]),
                    NUM_COLS), clust_idx);
#if VERBOSE
        BOOST_LOG_TRIVIAL(info) << "Dist to prev mean for c:" << clust_idx
            << " is " << cls->get_prev_dist(clust_idx);
#endif

        cluster_assignment_counts[clust_idx] = cls->get_num_members(clust_idx);
        chk_nmemb += cluster_assignment_counts[clust_idx];
    }
    BOOST_VERIFY(chk_nmemb == NUM_ROWS);

#if KM_TEST
    BOOST_LOG_TRIVIAL(info) << "Global number of changes: " << g_num_changed;
#endif
}

#if KM_TEST
void get_sampling(std::vector<std::vector<double>>& samples,
        const unsigned* cluster_assignments,
         const double* data, const size_t* cluster_assignment_counts) {
    constexpr unsigned MAX_PLOT_POINTS = 1000;
    const size_t samples_per_cluster =
        NUM_ROWS > MAX_PLOT_POINTS ? MAX_PLOT_POINTS : NUM_ROWS/K;

    if (samples_per_cluster < MAX_PLOT_POINTS) { // Just plot everything
        // TODO
    }

    for (unsigned k=0; k < K; k++) {
        //TODO: Add sample to the sampling result
    }
}
#endif
} // End annon namespace

namespace kpmeans { namespace omp {

kpmbase::kmeans_t compute_min_kmeans(const double* matrix, double* clusters_ptr,
        unsigned* cluster_assignments, size_t* cluster_assignment_counts,
        const size_t num_rows, const size_t num_cols, const unsigned k,
        const size_t MAX_ITERS, int max_threads, const std::string init,
        const double tolerance, const std::string dist_type) {
#ifdef PROFILER
    ProfilerStart("matrix/min-tri-kmeans.perf");
#endif
    NUM_COLS = num_cols;
    K = k;
    NUM_ROWS = num_rows;
    if (!max_threads)
        max_threads = 1;

    OMP_MAX_THREADS = std::min(max_threads, kpmbase::get_num_omp_threads());
    omp_set_num_threads(OMP_MAX_THREADS);
    BOOST_LOG_TRIVIAL(info) << "Running on " << OMP_MAX_THREADS << " threads!";

    // Check k
    if (K > NUM_ROWS || K < 2 || K == (unsigned)-1) {
        throw kpmbase::parameter_exception("'k' must be between 2 and"
                " the number of rows in the matrix.", K);
    }

    gettimeofday(&start , NULL);
    /*** Begin VarInit of data structures ***/
    std::fill(cluster_assignments, cluster_assignments+NUM_ROWS,
            kpmbase::INVALID_CLUSTER_ID);
    std::fill(cluster_assignment_counts, cluster_assignment_counts+K, 0);

    kpmbase::prune_clusters::ptr clusters =
        kpmbase::prune_clusters::create(K, NUM_COLS);

    if (init == "none")
        clusters->set_mean(clusters_ptr);

    // For pruning
    kpmbase::thd_safe_bool_vector::ptr recalculated_v =
        kpmbase::thd_safe_bool_vector::create(NUM_ROWS, false);

    std::vector<double> dist_v;
    dist_v.assign(NUM_ROWS, std::numeric_limits<double>::max());
    kpmprune::dist_matrix::ptr dm = kpmprune::dist_matrix::create(K);

    /*** End VarInit ***/
    BOOST_LOG_TRIVIAL(info) << "Dist_type is " << dist_type;
    if (dist_type == "eucl") {
        g_dist_type = kpmbase::dist_type_t::EUCL;
    } else if (dist_type == "cos") {
        g_dist_type = kpmbase::dist_type_t::COS;
    } else {
        throw kpmbase::parameter_exception("param `dist_type` must be one of: "
                "'eucl', 'cos'.", dist_type);
    }

    if (init == "random") {
        random_partition_init(cluster_assignments, matrix,
                clusters, NUM_ROWS, NUM_COLS, K);
        g_init_type = kpmbase::init_type_t::RANDOM;
        clusters->finalize_all();
    } else if (init == "forgy") {
        forgy_init(matrix, clusters, NUM_ROWS, NUM_COLS, K);
        g_init_type = kpmbase::init_type_t::FORGY;
    } else if (init == "kmeanspp") {
        kmeanspp_init(matrix, clusters, cluster_assignments);
        g_init_type = kpmbase::init_type_t::PLUSPLUS;
    } else if (init == "none") {
        g_init_type = kpmbase::init_type_t::NONE;
        dm->compute_dist(clusters, NUM_COLS);
    } else {
        throw kpmbase::parameter_exception("param `init` must be one of: "
            "'random', 'forgy', 'kmeanspp'", init);
    }

#if VERBOSE
    dm->compute_dist(clusters, NUM_COLS);
    BOOST_LOG_TRIVIAL(info) << "Cluster distance matrix after init ...";
    dm->print();
#endif

    BOOST_LOG_TRIVIAL(info) << "Init is '" << init << "'";

    if (MAX_ITERS > 0) {
        BOOST_LOG_TRIVIAL(info) << "Running INIT engine:";
        EM_step(matrix, clusters, cluster_assignments,
                cluster_assignment_counts, recalculated_v,
                dist_v, dm, true);
    }
#if KM_TEST
#ifndef BIND
        printf("Cluster assignment counts: ");
#endif
        print_arr(cluster_assignment_counts, K);
#endif

    g_num_changed = 0;
    BOOST_LOG_TRIVIAL(info) << "Matrix K-means starting ...";

    bool converged = false;
    std::string str_iters = MAX_ITERS == std::numeric_limits<size_t>::max() ?
        "until convergence ...":
        std::to_string(MAX_ITERS) + " iterations ...";
    BOOST_LOG_TRIVIAL(info) << "Computing " << str_iters;

    size_t iter = 0;
    if (MAX_ITERS > 0)
        iter = 2;

    while (iter < MAX_ITERS) {
        // Hold cluster assignment counter
        BOOST_LOG_TRIVIAL(info) << "E-step Iteration " << iter <<
            ". Computing cluster assignments ...";

#if VERBOSE
        BOOST_LOG_TRIVIAL(info) << "Main: Computing cluster distance matrix ...";
#endif
        dm->compute_dist(clusters, NUM_COLS);
#if VERBOSE
        BOOST_LOG_TRIVIAL(info) << "Before: Cluster distance matrix ...";
        dm->print();
#endif

        EM_step(matrix, clusters, cluster_assignments,
                cluster_assignment_counts, recalculated_v, dist_v, dm);
#if VERBOSE
        BOOST_LOG_TRIVIAL(info) << "Before: Printing clusters:";
        clusters->print_means();
#endif
#if KM_TEST
        BOOST_LOG_TRIVIAL(info) << "Printing cluster counts ...";
        print_arr(cluster_assignment_counts, K);
#endif

        if (g_num_changed == 0 || ((g_num_changed/(double)NUM_ROWS))
                <= tolerance) {
            converged = true;
            break;
        } else {
            g_num_changed = 0;
        }
        iter++;
    }

    gettimeofday(&end, NULL);
    BOOST_LOG_TRIVIAL(info) << "\n\nAlgorithmic time taken = " <<
        kpmbase::time_diff(start, end) << " sec\n";

#ifdef PROFILER
    ProfilerStop();
#endif
    BOOST_LOG_TRIVIAL(info) << "\n******************************************\n";

    if (converged) {
        BOOST_LOG_TRIVIAL(info) <<
            "K-means converged in " << iter << " iterations";
    } else {
        BOOST_LOG_TRIVIAL(warning) << "[Warning]: K-means failed to converge in "
            << iter << " iterations";
    }
    BOOST_LOG_TRIVIAL(info) << "Final cluster counts ...";
    kpmbase::print_arr(cluster_assignment_counts, K);
    BOOST_LOG_TRIVIAL(info) << "\n******************************************\n";

    return kpmbase::kmeans_t (NUM_ROWS, NUM_COLS, iter, K,
            cluster_assignments, cluster_assignment_counts,
            clusters->get_means());
}
} } // End namespace kpmeans, omp

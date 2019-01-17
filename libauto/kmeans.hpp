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

#ifndef __KNOR_OMP_KMEANS_HPP__
#define __KNOR_OMP_KMEANS_HPP__

#include <cstdlib>
#include <string>
#include "types.hpp"

namespace knor { namespace base {
    class cluster_t;
}}


namespace knor { namespace omp {
/**
 * \brief Compute kmeans on matrix of features
 * \param matrix The matrix who's row IDs are being clustered.
 * \param clusters The cluster centers (means).
 * \param cluster_assignments Which cluster each sample falls into.
 * \param cluster_assignment_counts How many members each cluster has.
 * \param num_rows The number of rows in `matrix`.
 * \param nev The number of eigenvalues / number of columns in `matrix`.
 * \param k The number of clusters required.
 * \param max_iters The maximum number of iterations of K-means to perform.
 * \param init The type of initilization ["random", "forgy", "kmeanspp"]
 **/
knor::base::cluster_t compute_kmeans(const double* matrix, double* clusters,
		unsigned* cluster_assignments, llong_t* cluster_assignment_counts,
		const size_t num_rows, const size_t num_cols, const unsigned k,
		const size_t MAX_ITERS, int max_threads,
        const std::string init="kmeanspp", const double tolerance=-1,
        const std::string dist_type="eucl");

/** See `compute_kmeans` for argument list */
knor::base::cluster_t compute_min_kmeans
    (const double* matrix, double* clusters_ptr,
        unsigned* cluster_assignments, llong_t* cluster_assignment_counts,
		const size_t num_rows, const size_t num_cols, const unsigned k,
        const size_t MAX_ITERS, int max_threads,
        const std::string init="kmeanspp", const double tolerance=-1,
        const std::string dist_type="eucl");
} }
#endif

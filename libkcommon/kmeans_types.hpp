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

#ifndef __KPM_TYPES_HPP__
#define __KPM_TYPES_HPP__

#include <cstddef>
#include <limits>
#include <vector>
#include <string>

namespace kpmeans { namespace base {

static const unsigned INVALID_CLUSTER_ID = std::numeric_limits<unsigned>::max();
enum kms_stage_t { INIT, ESTEP }; // What phase of the algo we're in
enum dist_type_t { EUCL, COS }; // Euclidean, Cosine distance
enum init_type_t { RANDOM, FORGY, PLUSPLUS, NONE }; // May have to use

class kmeans_t {
public:
    size_t nrow, ncol, iters, k;
    std::vector<unsigned> assignments;
    std::vector<size_t> assignment_count;
    std::vector<double> centroids;

    kmeans_t(){ }
    kmeans_t(const size_t nrow, const size_t ncol, const size_t iters,
             const size_t k, const unsigned* assignments_buf,
             const size_t* assignment_count_buf,
             const std::vector<double>& centroids);
    const void print() const;
    const void write(const std::string dirname) const;
    bool operator==(const kmeans_t& other);

    void set_params(const size_t nrow, const size_t ncol, const size_t iters,
             const size_t k) {
        this->nrow = nrow;
        this->ncol = ncol;
        this->iters = iters;
        this->k = k;
    };

    void set_computed(const unsigned* assignments_buf,
             const size_t* assignment_count_buf,
             const std::vector<double> centroids);

    ~kmeans_t() { }
};
} }
#endif

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

#ifndef __KNOR_TYPES_HPP__
#define __KNOR_TYPES_HPP__

#include <cstddef>
#include <limits>
#include <vector>
#include <string>
#include "dense_matrix.hpp"

namespace knor {

    typedef long long int llong_t;
    namespace base {

static const unsigned INVALID_CLUSTER_ID = std::numeric_limits<unsigned>::max();
enum stage_t { INIT, ESTEP }; // What phase of the algo we're in
enum dist_t { EUCL, COS, TAXI, SQEUCL }; // Euclidean, Cosine, Taxicab distance
enum init_t { RANDOM, FORGY, PLUSPLUS, NONE }; // May have to use

class cluster_t {
public:
    size_t nrow, ncol, iters, k;
    std::vector<unsigned> assignments;
    std::vector<size_t> assignment_count;
    std::vector<double> centroids;

    cluster_t(){ }
    cluster_t(const size_t nrow, const size_t ncol, const size_t iters,
             const size_t k, const unsigned* assignments_buf,
             const llong_t* assignment_count_buf,
             const std::vector<double>& centroids);
    const void print() const;
    const void write(const std::string dirname) const;
    bool operator==(const cluster_t& other);

    void set_params(const size_t nrow, const size_t ncol, const size_t iters,
             const size_t k) {
        this->nrow = nrow;
        this->ncol = ncol;
        this->iters = iters;
        this->k = k;
    };

    void set_computed(const unsigned* assignments_buf,
             const llong_t* assignment_count_buf,
             const std::vector<double> centroids);

    ~cluster_t() { }
};

struct gmm_t {
    size_t nrow, ncol, iters;
    unsigned k;
    std::vector<double> means;
    std::vector<std::vector<double> > cov_mats;
    std::vector<double> resp_mat;
    std::vector<double> gaussian_prob;

    gmm_t() { }
    bool operator==(const gmm_t& other);
    gmm_t(const size_t nrow, const size_t ncol, const size_t iters,
            const size_t k, double* _means,
            std::vector<base::dense_matrix<double>*>& _cov_mats,
            double* _resp_mat, double* _gaussian_prob);
};
} }
#endif

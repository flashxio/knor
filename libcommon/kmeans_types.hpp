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

#include <limits>
#include <vector>
#include "io.hpp"

namespace kpmeans { namespace base {

static const unsigned INVALID_CLUSTER_ID = std::numeric_limits<unsigned>::max();
enum kms_stage_t { INIT, ESTEP }; // What phase of the algo we're in
enum dist_type_t { EUCL, COS }; // Euclidean, Cosine distance
enum init_type_t { RANDOM, FORGY, PLUSPLUS, NONE }; // May have to use

template <typename T>
class kmeans_t {
public:
    std::vector<unsigned> assignments;
    std::vector<size_t> assignment_count;
    size_t iters;
    std::vector<T> centroids;

    kmeans_t(std::vector<unsigned>& assignments,
            size_t* assignment_count_buf, const size_t k,
            const size_t iters, std::vector<T>& centroids) {
        this->assignments = assignments;
        this->iters = iters;
        this->assignment_count.resize(k);
        std::copy(assignment_count_buf, assignment_count_buf + k,
                assignment_count.begin());
        this->centroids = centroids;
    }

    const void print() const {
        std::cout << "Iterations: " <<  iters << std::endl;
        std::cout << "Cluster count: ";
        kpmeans::base::print_vector(assignment_count);
    }
};
} }
#endif

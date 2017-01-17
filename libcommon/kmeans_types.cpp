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

#include "kmeans_types.hpp"
#include "io.hpp"
#include "util.hpp"

namespace kpmeans { namespace base {

kmeans_t::kmeans_t(const size_t nrow, const size_t ncol, const size_t iters,
         const size_t k, const unsigned* assignments_buf,
         const size_t* assignment_count_buf,
         const std::vector<double>& centroids) {
    this->nrow = nrow;
    this->ncol = ncol;
    this->iters = iters;

    assignment_count.resize(k);
    assignments.resize(nrow);

    std::copy(assignments_buf, assignments_buf + nrow, assignments.begin());
    std::copy(assignment_count_buf, assignment_count_buf + k,
            assignment_count.begin());
    this->centroids = centroids; // copy
}

const void kmeans_t::print() const {
    std::cout << "Iterations: " <<  iters << std::endl;
    std::cout << "Cluster count: ";
    print_vector(assignment_count);
}

bool kmeans_t::operator==(const kmeans_t& other) {
    return (v_eq(this->assignments, other.assignments) &&
            v_eq(this->assignment_count, other.assignment_count) &&
            v_eq(this->centroids, other.centroids));

};
} }

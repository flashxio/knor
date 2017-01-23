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

#include <fstream>

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
    this->k = k;

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

/**
  * A simple text readable write
  * \param dirname: the name of the dir to write to
  */
const void kmeans_t::write(const std::string dirname) const {

    std::string fn = "kmeans_t.txt";
    int ret =
        std::system((std::string("python exec/python/util.py ")
                    + dirname).c_str());
    if (ret)
        fprintf(stderr, "Error with mkdir. Code: %d\n", ret);
    else
        fn = dirname + "/" + fn;

    printf("Opening '%s' \n", fn.c_str());
    std::ofstream f(fn, std::ios::out);
    BOOST_ASSERT_MSG(f.is_open(), "Error opening file for writing!");
    f << "k: " << k << std::endl;
    f << "niter: " << iters << std::endl;
    f << "nrow: " << nrow << std::endl;
    f << "ncol: " << ncol << std::endl;

    /** Do it all here so we can stream the output **/
    // Sizes of each cluster
    f << "size:\n";
    for (size_t _k = 0; _k < k; _k++) {
        if (_k == 0)
            f << assignment_count[_k];
        else
            f << " " << assignment_count[_k];
    }

    // Centroid assignemnt
    f << "\ncluster:\n";
    for (size_t row = 0; row < nrow; row++) {
        if (row == 0)
            f << assignments[row];
        else
            f << " " << assignments[row];
    }

    // Centroids
    f << "\ncentroids:";
    for (size_t row = 0; row < k; row++) {
        for (size_t col = 0; col < ncol; col++) {
            if (col == 0)
                f << "\n" << centroids[row*ncol+col];
            else
                f << " " << centroids[row*ncol+col];
        }
    }

    f << std::endl;

    f.close();
}

bool kmeans_t::operator==(const kmeans_t& other) {
    return (v_eq(this->assignments, other.assignments) &&
            v_eq(this->assignment_count, other.assignment_count) &&
            v_eq(this->centroids, other.centroids));

};
} }

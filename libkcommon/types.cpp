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

#include <fstream>

#include "types.hpp"
#include "io.hpp"
#include "util.hpp"
#include "hclust_compactor.hpp"

namespace kutil = knor::util;

namespace knor { namespace base {

cluster_t::cluster_t(const size_t nrow, const size_t ncol, const size_t iters,
         const size_t k, const unsigned* assignments_buf,
         const llong_t* assignment_count_buf,
         const std::vector<double>& centroids) : nrow(nrow), ncol(ncol),
    iters(iters), k(k) {

    assignment_count.resize(k);
    assignments.resize(nrow);

    std::copy(assignments_buf, assignments_buf + nrow, assignments.begin());
    std::copy(assignment_count_buf, assignment_count_buf + k,
            assignment_count.begin());
    this->centroids = centroids; // copy
}

cluster_t::cluster_t(const size_t nrow, const size_t ncol, const size_t iters,
             const std::vector<unsigned>& assignments_buf,
     std::vector<llong_t>& assignment_count_buf,
     const std::unordered_map<unsigned, std::vector<double>>& centroids)
    : nrow(nrow), ncol(ncol), iters(iters) {

    assignments.resize(nrow);

    std::unordered_map<unsigned, unsigned> id_map;
    // Populates assignment_count
    kutil::compactor::remap<llong_t>(assignment_count_buf,
            assignment_count, id_map);

    // Reassign elements
    for (size_t i = 0; i < assignments_buf.size(); i++)
        assignments[i] = id_map[assignments_buf[i]];

    this->centroids.resize(centroids.size()*ncol);

    for (auto const& kv : centroids) {
        auto const& k = kv.first;
        auto const& v = kv.second;
        std::copy(v.cbegin(), v.cend(), &(this->centroids[id_map[k]*ncol]));
    }

    k = centroids.size();
}

void cluster_t::set_computed(const unsigned* assignments_buf,
        const llong_t* assignment_count_buf,
        const std::vector<double> centroids) {

    assert(this->k);
    assert(this->nrow);
    assignment_count.resize(this->k);
    assignments.resize(this->nrow);

    std::copy(assignments_buf, assignments_buf + nrow, assignments.begin());
    std::copy(assignment_count_buf, assignment_count_buf + k,
            assignment_count.begin());
    this->centroids = centroids; // copy
}

const void cluster_t::print() const {
#ifndef BIND
    std::cout << "nrow: " << nrow << ", ncol: " << ncol <<
    ", iters: " <<  iters << ", k: " << k << std::endl;
    std::cout << "Assignment count:\n";
    knor::base::print(assignment_count);
    std::cout << "Assignment: \n";
    knor::base::print(assignments);
    std::cout << "Centroids: \n";
    knor::base::print(&centroids[0], k, ncol);;
#endif
}

/**
  * A simple text readable write
  * \param dirname: the name of the dir to write to
  */
const void cluster_t::write(const std::string dirname) const {

    std::string fn = "cluster_t.yml";
    int ret =
        std::system((std::string("python python/util.py ")
                    + dirname).c_str());
    if (ret) {
#ifndef BIND
        fprintf(stderr, "Error with mkdir. Code: %d\n", ret);
#endif
    } else {
        fn = dirname + "/" + fn;
    }

#ifndef BIND
    printf("Opening '%s' \n", fn.c_str());
#endif
    std::ofstream f(fn, std::ios::out);
    //assert_msg(f.is_open(), "Error opening file for writing!");
    f << "k: " << k << std::endl;
    f << "niter: " << iters << std::endl;
    f << "nsamples: " << nrow << std::endl;
    f << "dim: " << ncol << std::endl;

    /** Do it all here so we can stream the output **/
    // Sizes of each cluster
    f << "size: [";
    for (size_t _k = 0; _k < k; _k++) {
        if (_k == 0)
            f << assignment_count[_k];
        else
            f << "," << assignment_count[_k];
    }
    f << "]";

    // Centroid assignemnt
    f << "\ncluster: [";
    for (size_t row = 0; row < nrow; row++) {
        if (row == 0)
            f << assignments[row];
        else
            f << "," << assignments[row];
    }
    f << "]";

    // Centroids
    f << "\ncentroids: [";
    for (size_t row = 0; row < k; row++) {
        if (row == 0)
            f << "[";
        else
            f << ",[";
        for (size_t col = 0; col < ncol; col++) {
            if (col == 0)
                f << centroids[row*ncol+col];
            else
                f << "," << centroids[row*ncol+col];
        }
        f << "]";
    }

    f << "]\n";
    f.close();
}

const bool cluster_t::operator==(const cluster_t& other) const {
    return (v_eq(this->assignments, other.assignments) &&
            v_eq(this->assignment_count, other.assignment_count) &&
            v_eq(this->centroids, other.centroids));
}

const std::string cluster_t::to_str() {
    std::string s = "Iterations: ";
    s += std::to_string(iters);
    s += "\nk: ";
    s += std::to_string(k);
    s += "\nnrow: ";
    s += std::to_string(nrow);
    s += "\nncol: ";
    s += std::to_string(ncol);
    s += "\nCluster count: \n[ ";

    for (const auto& v : assignment_count) {
        s += std::to_string(v);
        s += " ";
    }

    s += "]\n";
    return s;
}

gmm_t::gmm_t(const size_t nrow, const size_t ncol, const size_t iters,
        const size_t k, double* _means,
        std::vector<base::dense_matrix<double>*>& _cov_mats,
        double* _resp_mat, double* _gaussian_prob) : nrow(nrow),
    ncol(ncol), iters(iters), k(k) {

        means.resize(k*ncol);
        std::copy(&(_means[0]), &(_means[k*ncol]), means.begin());
        for (auto const& dm : _cov_mats) {
            std::vector<double> v(ncol*ncol);
            std::copy(dm->as_vector().begin(), dm->as_vector().end(), v.begin());
            cov_mats.push_back(v);
        }

        resp_mat.resize(nrow);
        std::copy(&(_resp_mat[0]), &(_resp_mat[nrow]), resp_mat.begin());

        gaussian_prob.resize(k);
        std::copy(&(_gaussian_prob[0]), &(_gaussian_prob[k]),
                gaussian_prob.begin());
    }

bool gmm_t::operator==(const gmm_t& other) {

    for (size_t i = 0; i < cov_mats.size(); i++)
        if (!v_eq(cov_mats[i], other.cov_mats[i]))
            return false;

    return nrow == other.nrow && ncol == other.ncol &&
            iters == other.iters && k == other.k &&
            v_eq(this->means, other.means) &&
            v_eq(this->resp_mat, other.resp_mat) &&
            v_eq(this->gaussian_prob, other.gaussian_prob);
}

}} // End namespace knor::base

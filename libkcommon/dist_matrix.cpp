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

#include <cassert>
#include <iostream>

#include "dist_matrix.hpp"
#include "clusters.hpp"
#include "io.hpp"
#include "util.hpp"

namespace knor { namespace prune {

dist_matrix::dist_matrix(const unsigned rows) {
    assert(rows > 1);

    this->rows = rows-1;
    // Distance to everyone other than yourself
    for (unsigned i = this->rows; i > 0; i--) {
        std::vector<double> dist_row;
        dist_row.assign(i, std::numeric_limits<double>::max());
        mat.push_back(dist_row);
    }
}

void dist_matrix::translate(unsigned& row, unsigned& col) {
    // First make sure the smaller is the row
    if (row > col) std::swap(row, col);

    assert(row < rows);
    col = col - row - 1; // Translation
    assert(col < (rows - row));
}

/* Do a translation from raw id's to indexes in the distance matrix */
double dist_matrix::get(unsigned row, unsigned col) {
    if (row == col) { return std::numeric_limits<double>::max(); }
    translate(row, col);
    return mat[row][col];
}

// Testing purposes only
double dist_matrix::get_min_dist(const unsigned row) {
    double best = std::numeric_limits<double>::max();
    for (unsigned col = 0; col < rows+1; col++) {
        if (col != row) {
            double val = get(row, col);
            if (val < best) best = val;
        }
    }
    assert(best < std::numeric_limits<double>::max());
    return best;
}


void dist_matrix::set(unsigned row, unsigned col, double val) {
    assert(row != col);
    translate(row, col);
    mat[row][col] = val;
}

void dist_matrix::print() {
    for (unsigned row = 0; row < rows; row++) {
#ifndef BIND
        std::cout << row << " ==> ";
#endif
        knor::base::print<double>(mat[row]);
    }
}

void dist_matrix::compute_dist(knor::base::prune_clusters::ptr cls,
        const unsigned ncol) {
    if (cls->get_nclust() <= 1) return;

    assert(get_num_rows() == cls->get_nclust()-1);
    cls->reset_s_val_v();
#ifdef _OPENMP
    //#pragma omp parallel for collapse(2) // FIXME: Opt Coalese perhaps
#endif
    for (unsigned i = 0; i < cls->get_nclust(); i++) {
        for (unsigned j = i+1; j < cls->get_nclust(); j++) {
            double dist = knor::base::eucl_dist(&(cls->get_means()[i*ncol]),
                    &(cls->get_means()[j*ncol]), ncol) / 2.0;
            set(i,j, dist);

            // Set s(x) for each cluster
            if (dist < cls->get_s_val(i)) {
                cls->set_s_val(dist, i);
            }

            if (dist < cls->get_s_val(j)) {
                cls->set_s_val(dist, j);
            }
        }
    }
#if VERBOSE
    for (unsigned cl = 0; cl < cls->get_nclust(); cl++) {
        assert(cls->get_s_val(cl) == get_min_dist(cl));
#ifndef BIND
        printf("cl: %u get_s_val: %.6f\n", cl, cls->get_s_val(cl));
#endif
    }
#endif
}

// Used for PAM pairwise distance of all entries
// TODO: Find soln to this for Mac
void dist_matrix::compute_pairwise_dist(double* data,
        const size_t ncol, const knor::base::dist_t metric) {
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = i+1; j < rows+1; j++) {
            switch (metric) {
                case (knor::base::dist_t::EUCL):
                    {
                        double dist = knor::base::eucl_dist(&(data[i*ncol]),
                                &(data[j*ncol]), ncol);
                        set(i,j, dist);
                        break;
                    }
                case (knor::base::dist_t::COS):
                    {
                        double dist = knor::base::cos_dist(&(data[i*ncol]),
                                &(data[j*ncol]), ncol);
                        set(i,j, dist);
                        break;
                    }
                case (knor::base::dist_t::TAXI):
                    {
                        double dist = knor::base::taxi_dist<double>
                            (&(data[i*ncol]), &(data[j*ncol]), ncol);
                        set(i,j, dist);
                        break;
                    }
                default:
                    throw knor::base::parameter_exception(
                            "Unknown distance metric");
            }
        }
    }
}
} } // End namepsace knor, prune
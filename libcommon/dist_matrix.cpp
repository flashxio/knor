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

#include "common.hpp"

namespace kpmeans { namespace prune {

dist_matrix::dist_matrix(const unsigned rows) {
    BOOST_VERIFY(rows > 1);

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
    if (row > col) {
        std::swap(row, col);
    }

    BOOST_VERIFY(row < rows);
    col = col - row - 1; // Translation
    BOOST_VERIFY(col < (rows - row));
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
    BOOST_VERIFY(best < std::numeric_limits<double>::max());
    return best;
}


void dist_matrix::set(unsigned row, unsigned col, double val) {
    BOOST_VERIFY(row != col);
    translate(row, col);
    mat[row][col] = val;
}

void dist_matrix::print() {
    for (unsigned row = 0; row < rows; row++) {
        std::cout << row << " ==> ";
        kpmeans::base::print_vector<double>(mat[row]);
    }
}
} } // End namepsace kpmeans, prune

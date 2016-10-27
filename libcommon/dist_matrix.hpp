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

#ifndef __KPM_DIST_MATRIX_HPP__
#define __KPM_DIST_MATRIX_HPP__

#include <memory>
#include <limits>
#include <vector>

#include <El.hpp>

#include "util.hpp"

namespace kpmbase = kpmeans::base;

namespace kpmeans { namespace prune {
// NOTE: Creates a matrix like this e.g for K = 5
/* - Don't store full matrix, don't store dist to myself -> space: (k*k-1)/2
   0 ==> 1 2 3 4
   1 ==> 2 3 4
   2 ==> 3 4
   3 ==> 4
   (4 ==> not needed)
   */
class dist_matrix {
private:
    std::vector<std::vector<double>> mat;
    unsigned rows;

    dist_matrix(const unsigned rows);
    void translate(unsigned& row, unsigned& col);

public:
    typedef typename std::shared_ptr<dist_matrix> ptr;

    static ptr create(const unsigned rows) {
        return ptr(new dist_matrix(rows));
    }

    const unsigned get_num_rows() { return rows; }

    /* Do a translation from raw id's to indexes in the distance matrix */
    double get(unsigned row, unsigned col);
    // Testing purposes only
    double get_min_dist(const unsigned row);
    void set(unsigned row, unsigned col, double val);

    void print();
    // Note cls is col-wise unlike dist_matrix
    template<typename T>
    void compute_dist(El::Matrix<T>& cls,
            std::vector<T>& s_val_v) {
    if (cls.Width() == 1) {
        s_val_v.push_back((T)0);
        return;
    }

    BOOST_VERIFY(get_num_rows()+1 == (El::Unsigned)cls.Width());
    std::fill(s_val_v.begin(), s_val_v.end(),
            std::numeric_limits<double>::max());

    for (El::Unsigned i = 0; i < (El::Unsigned)cls.Width(); i++) {
        for (El::Unsigned j = i+1; j < (El::Unsigned)cls.Width(); j++) {
            double dist = kpmbase::eucl_dist(cls.LockedBuffer(0,i),
                    cls.LockedBuffer(0,j), cls.Height()) / 2.0;

            set(i,j, dist);

            // Set s(x) for each cluster
            if (dist < s_val_v[i])
                s_val_v[i] = dist;

            if (dist < s_val_v[j])
                s_val_v[j] = dist;
        }
    }
#if VERBOSE
    for (El::Unsigned cl = 0; cl < (El::Unsigned)cls.Width(); cl++) {
        BOOST_VERIFY(s_val_v[cl] == get_min_dist(cl));
        if (El::mpi::Rank(El::mpi::COMM_WORLD) == 0)
            El::Output("cl: ", cl," get_s_val: ", s_val_v[cl]);
    }
#endif
}
};
} } // End namespace kpmeans, prune
#endif

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

#ifndef __KNOR_DENSE_MATRIX_HPP__
#define __KNOR_DENSE_MATRIX_HPP__

#include <vector>
#include "io.hpp"

namespace knor { namespace base {

template <typename T>
class dense_matrix {
private:
    std::vector<T> mat;
    size_t nrow;
    size_t ncol;

public:
    typedef dense_matrix* rawptr;

    dense_matrix(const size_t nrow, const size_t ncol) :
    nrow(nrow), ncol(ncol) {
        mat.resize(nrow*ncol);
    }

    static rawptr create(const size_t nrow, const size_t ncol) {
        return new dense_matrix(nrow, ncol);
    }

    const size_t get_nrow() const { return nrow; }
    const size_t get_ncol() const { return ncol; }
    void set_nrow(const size_t nrow) { this->nrow = nrow; }
    void set_ncol(const size_t ncol) { this->ncol = ncol; }

    void set(const T* d) {
        std::copy(&(d[0]), &(d[nrow*ncol]), mat.begin());
    }

    void set_row(const T* d, const size_t rid) {
        std::copy(&(d[0]), &(d[ncol]), &(mat[rid*ncol]));
    }

    /* Do a translation from raw id's to indexes in the distance matrix */
    const T get(const size_t row, const size_t col) {
        return mat[row*ncol+col];
    }

    void set(const size_t row, const size_t col, const T val) {
        mat[(row*ncol)+col] = val;
    }

    std::vector<T>& as_vector() { return mat; }
    T* as_pointer() { return &mat[0]; }

    void print() {
        print_mat<T>(as_pointer(), nrow, ncol);
    }
};
} } // End namespace knor, base
#endif

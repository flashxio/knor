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
#include "exception.hpp"
#include <omp.h>

namespace knor { namespace base {

// A rowmajor dense matrix
template <typename T>
class dense_matrix {
private:
    std::vector<T> mat;
    size_t nrow;
    size_t ncol;

public:
    typedef dense_matrix* rawptr;

    dense_matrix(const size_t nrow, const size_t ncol, bool zeros=false) :
    nrow(nrow), ncol(ncol) {
        if (zeros)
            mat.assign(nrow*ncol, 0);
        else
            mat.resize(nrow*ncol);
    }

    static rawptr create(const size_t nrow, const size_t ncol,
            bool zeros=false) {
        return new dense_matrix(nrow, ncol, zeros);
    }

    static rawptr create(dense_matrix<T>* other) {
        rawptr ret = new dense_matrix(other->get_nrow(), other->get_ncol());
        std::copy(other->as_pointer(),
                other->as_pointer()+(other->get_nrow()*other->get_ncol()),
                ret->as_pointer());
        return ret;
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
    const T& get(const size_t row, const size_t col) {
        return mat[row*ncol+col];
    }

    void set(const size_t row, const size_t col, const T val) {
        mat[(row*ncol)+col] = val;
    }

    std::vector<T>& as_vector() { return mat; }
    T* as_pointer() { return &mat[0]; }

    // dim = 0 is row wise mean
    // dim = 1 is col wise mean
    void mean(std::vector<double>& mean, const size_t dim=1) {
        if (dim == 1) {
            mean.assign(ncol, 0);
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    mean[col] += mat[row*ncol+col];
                }
            }

            for (size_t i = 0; i < mean.size(); i++)
                mean[i] /= (double)nrow;
        } else if (dim == 0) {
            mean.assign(nrow, 0);
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    mean[row] += mat[row*ncol+col];
                }
            }

            for (size_t i = 0; i < mean.size(); i++)
                mean[i] /= (double)ncol;
        }
    }

    dense_matrix<T>* operator-(std::vector<T>& v) {
        dense_matrix* ret = create(this);
        T* retp = ret->as_pointer();

        if (v.size() == nrow) {
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    retp[row*ncol+col] -= v[row];
                }
            }
        } else if (v.size() == ncol) {
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    retp[row*ncol+col] -= v[col];
                }
            }
        } else {
            throw parameter_exception("vector size is neither == nrow / ncol");
        }

        return ret;
    }

    dense_matrix* operator*(dense_matrix<T>& other) {
        dense_matrix<T>* res = create(this->nrow, other.get_ncol(), true);
        double* rp = res->as_pointer();

        // lhs is this object and rhs is other
#pragma omp parallel for
        for (size_t lrow = 0; lrow < this->nrow; lrow++) {
            printf("Thread %d working ...\n", omp_get_thread_num());
            for (size_t rrow = 0; rrow < other.get_nrow(); rrow++) {
                for (size_t rcol = 0; rcol < other.get_ncol(); rcol++) {
                    rp[lrow*other.get_ncol()+rcol] +=
                        mat[lrow*ncol+rrow] *
                        other.get(rrow, rcol);
                }
            }
        }
        return res;
    }

    dense_matrix& operator/=(const T val) {
        for (size_t i = 0; i < nrow*ncol; i++)
            mat[i] /= val;
        return *this;
    }

    bool operator==(dense_matrix<T>& other) {
        if (nrow != other.get_nrow() || ncol != other.get_ncol())
            return false;

        for (size_t i = 0; i < ncol*nrow; i++) {
            if (mat[i] != other.as_pointer()[i])
                return false;
        }
        return true;
    }

    void print() {
        print_mat<T>(as_pointer(), nrow, ncol);
    }
};
} } // End namespace knor, base
#endif

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

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "io.hpp"
#include "exception.hpp"

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

    dense_matrix() { nrow = 0; ncol = 0; }

    dense_matrix(const size_t nrow, const size_t ncol, bool zeros=false) :
    nrow(nrow), ncol(ncol) {
        if (zeros)
            mat.assign(nrow*ncol, 0);
        else
            mat.resize(nrow*ncol);
    }

    static rawptr create() {
        return new dense_matrix();
    }

    static rawptr create(const size_t nrow, const size_t ncol,
            bool zeros=false) {
        return new dense_matrix(nrow, ncol, zeros);
    }

    static rawptr create(dense_matrix* other) {
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

    dense_matrix* operator-(std::vector<T>& v) {
        assert(nrow == ncol);

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

    dense_matrix* operator*(dense_matrix& other) {
        dense_matrix* res = create(this->nrow, other.get_ncol(), true);
        double* rp = res->as_pointer();

        // lhs is this object and rhs is other
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t lrow = 0; lrow < this->nrow; lrow++) {
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

    dense_matrix* operator-(dense_matrix& other) {
        assert(nrow == other.get_nrow() && ncol == other.get_ncol());

        dense_matrix* res = create(nrow, ncol);
        double* rp = res->as_pointer();
        double* otherp = other.as_pointer();

        for (size_t i = 0; i < nrow*ncol; i++)
            rp[i] = mat[i] - otherp[i];
        return res;
    }

    dense_matrix* operator+(dense_matrix& other) {
        assert(nrow == other.get_nrow() && ncol == other.get_ncol());

        dense_matrix* res = create(nrow, ncol);
        double* rp = res->as_pointer();
        double* otherp = other.as_pointer();

        for (size_t i = 0; i < nrow*ncol; i++)
            rp[i] = mat[i] + otherp[i];
        return res;
    }

    void operator+=(dense_matrix& other) {
        assert(nrow == other.get_nrow() && ncol == other.get_ncol());
        double* otherp = other.as_pointer();

        for (size_t i = 0; i < nrow*ncol; i++)
            mat[i] += otherp[i];
    }

    void peq(const size_t row, size_t col, T val) {
        mat[row*ncol+col] += val;
    }

    T frobenius_norm() {
        T sum = 0;
        for (size_t i = 0; i < nrow*ncol; i++)
            sum += mat[i]*mat[i];
        return std::sqrt(sum);
    }

    void zero() {
        std::fill(mat.begin(), mat.end(), 0);
    }

    inline void sum(const unsigned axis, std::vector<T>& res) {
        // column wise
        if (axis == 0) {
            // Copy the first row
            res.resize(ncol);
            std::copy(mat.begin(), mat.begin()+ncol, res.begin());

            // TODO: Bottleneck
            for (size_t row = 1; row < nrow; row++) {
                size_t row_offset = row * ncol;
                for (size_t col = 0; col < ncol; col++) {
                    res[col] += mat[row_offset+col];
                }
            }
        } else if (axis == 1) { /* row wise */
            res.assign(nrow, 0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    res[row] += mat[row*ncol+col];
                }
            }
        } else {
            throw parameter_exception("axis for sum must be 0 or 1");
        }
    }

    T sum() {
        T sum = 0;
        for (size_t i = 0; i < nrow*ncol; i++)
            sum += mat[i];
        return sum;
    }

    dense_matrix& operator/=(const T val) {
        for (size_t i = 0; i < nrow*ncol; i++)
            mat[i] /= val;
        return *this;
    }

    dense_matrix& operator/=(std::vector<T>& v) {
        if (nrow == ncol)
            throw parameter_exception("Cannot determine which axis "
                    "to div for square matrix.");

        if (v.size() == nrow) {
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    mat[row*ncol+col] /= v[row];
                }
            }
        } else if (v.size() == ncol) {
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    mat[row*ncol+col] /= v[col];
                }
            }
        } else {
            throw std::runtime_error("Vector division must have size = nrow/ncol");
        }
        return *this;
    }

    void div_eq(std::vector<T>& v, const unsigned axis) {
        if (axis == 0 && v.size() == ncol) {
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    mat[row*ncol+col] /= v[col];
                }
            }
        } else if (axis == 1 && v.size() == nrow) {
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    mat[row*ncol+col] /= v[row];
                }
            }
        } else {
            throw std::runtime_error("Vector division must have size = nrow/ncol");
        }
    }

    void div_eq_pow(std::vector<T>& v, const unsigned axis,
            const unsigned exp) {

        if (axis == 0 && v.size() == ncol) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (size_t row = 0; row < nrow; row++) {
                //printf("Thread %d working\n", omp_get_thread_num());
                size_t row_offset = row * ncol;
                for (size_t col = 0; col < ncol; col++) {
                    mat[row_offset+col] =
                        std::pow(mat[row_offset+col] / v[col], exp);
                }
            }
        } else if (axis == 1 && v.size() == nrow) {
            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    mat[row*ncol+col] =
                        std::pow(mat[row*ncol+col] / v[row], exp);
                }
            }
        } else {
            throw std::runtime_error(
                    "Vector division must have size = nrow/ncol");
        }
    }

    // Raise every element to the power `exp` and assign
    void pow_eq(T exp) {
        for (size_t i = 0; i < nrow*ncol; i++)
            mat[i] = std::pow(mat[i], exp);
    }

    bool operator==(dense_matrix& other) {
        if (nrow != other.get_nrow() || ncol != other.get_ncol())
            return false;

        for (size_t i = 0; i < ncol*nrow; i++) {
            if (mat[i] != other.as_pointer()[i])
                return false;
        }
        return true;
    }

    void copy_from(dense_matrix* dm) {
        resize(dm->get_nrow(), dm->get_ncol());
        std::copy(dm->as_pointer(),
                dm->as_pointer()+(dm->get_nrow()*dm->get_ncol()),
                mat.begin());
    }

    void argmax(size_t axis, std::vector<unsigned>& idx) {
        std::vector<T> vals;

        // axis = 1: row wise
        if (axis == 0) {
            idx.assign(nrow, 0);
            vals.assign(nrow, std::numeric_limits<double>::min());

            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    T val = mat[row*ncol+col];
                    if (val > vals[row]) {
                        vals[row] = val;
                        idx[row] = col;
                    }
                }
            }
        // axis = 0: col wise
        } else if (axis == 1) {
            idx.assign(ncol, 0);
            vals.assign(ncol, std::numeric_limits<double>::min());

            for (size_t row = 0; row < nrow; row++) {
                for (size_t col = 0; col < ncol; col++) {
                    T val = mat[row*ncol+col];
                    if (val > vals[col]) {
                        vals[col] = val;
                        idx[col] = row;
                    }
                }
            }
        }  else {
            throw parameter_exception("axis for argmax must be 0 or 1");
        }
    }

    void resize(const size_t nrow, const size_t ncol) {
        mat.resize(nrow*ncol);
        this->nrow = nrow;
        this->ncol = ncol;
    }

    void print() {
        knor::base::print<T>(as_pointer(), nrow, ncol);
    }
};
} } // End namespace knor, base
#endif

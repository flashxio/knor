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
#ifndef __KPM_UTIL_HPP__
#define __KPM_UTIL_HPP__

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <vector>
#include <iostream>

#include <boost/assert.hpp>
#include <boost/log/trivial.hpp>
#include "kmeans_types.hpp"

namespace kpmeans { namespace base {

double get_bic(const std::vector<double>& dist_v, const unsigned nrow,
        const unsigned ncol, const unsigned k);
void spherical_projection(double* data, const unsigned nrow,
        const unsigned ncol);

// Vector equal function
template <typename T>
bool v_eq(const T& lhs, const T& rhs) {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T>
const bool v_eq_const(const std::vector<T>& v, const T var) {
    for (unsigned i=0; i < v.size(); i++) {
        if (v[i] != var) return false;
    }
    return true;
}

template <typename T>
bool eq_all(const T* v1, const T* v2, const unsigned len) {
    return (std::equal(&v1[0], &(v1[len-1]), &v2[0]));
}

template <typename T>
const double eucl_dist(const T* lhs, const T* rhs,
        const unsigned size) {
    double dist = 0;
    double diff;

    for (unsigned col = 0; col < size; col++) {
        diff = lhs[col] - rhs[col];
        dist += diff * diff;
    }
    return sqrt(dist);
}

template<typename T>
const double cos_dist(const T* lhs, const T* rhs,
        const unsigned size) {
    T numr, ldenom, rdenom;
    numr = ldenom = rdenom = 0;

    for (unsigned col = 0; col < size; col++) {
        T a = lhs[col];
        T b = rhs[col];

        numr += a*b;
        ldenom += a*a;
        rdenom += b*b;
    }
    return  1 - (numr / ((sqrt(ldenom)*sqrt(rdenom))));
}

/** /brief Choose the correct distance function and return it
 * /param arg0 A pointer to data
 * /param arg1 Another pointer to data
 * /param len The number of elements used in the comparison
 * /return the distance based on the chosen distance metric
 */
template <typename T>
T dist_comp_raw(const T* arg0, const T* arg1,
        const unsigned len, dist_type_t dt) {
    if (dt == dist_type_t::EUCL)
        return eucl_dist<T>(arg0, arg1, len);
    else if (dt == dist_type_t::COS)
        return cos_dist(arg0, arg1, len);
    else
        BOOST_ASSERT_MSG(false, "Unknown distance metric!");
    exit(EXIT_FAILURE);
}

float time_diff(struct timeval time1, struct timeval time2);
int get_num_omp_threads();

} } // End namespace kpmeans::base
#endif

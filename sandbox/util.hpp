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
#include <vector>

namespace kpmeans { namespace base {

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

} } // End namespace kpmeans, base
#endif

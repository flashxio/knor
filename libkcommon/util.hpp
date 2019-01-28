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
#ifndef __KNOR_UTIL_HPP__
#define __KNOR_UTIL_HPP__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <vector>
#include <iostream>
#include <random>

#include "types.hpp"
#include "exception.hpp"

namespace knor { namespace base {

double get_bic(const std::vector<double>& dist_v, const size_t nrow,
        const size_t ncol, const unsigned k);
void spherical_projection(double* data, const size_t nrow,
        const size_t ncol);

// Hierarchical ceiling of the number of clusters you can get given some non
//  power of two
unsigned get_hclust_ceil(const unsigned k);

// Hierarchical floor of the number of clusters you can get given some non
//  power of two
unsigned get_hclust_floor(const unsigned k);

template <typename T>
T get_max_hnodes(const T v) {
    T nodes = 0;
    for (T i = 1; i <= v; i*=2)
        nodes += i;
    return nodes;
}

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
    return std::sqrt(dist);
}

template <typename T>
const double sqeucl_dist(const T* lhs, const T* rhs,
        const unsigned size) {
    double dist = 0;
    double diff;

    for (unsigned col = 0; col < size; col++) {
        diff = lhs[col] - rhs[col];
        dist += diff * diff;
    }
    return dist;
}

template <typename T>
const T taxi_dist(const T* lhs, const T* rhs,
        const unsigned size) {
    T dist = 0;

    for (unsigned col = 0; col < size; col++) {
        dist += std::abs(lhs[col] - rhs[col]);
    }
    return dist;
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
    return  1 - (numr / ((std::sqrt(ldenom)*std::sqrt(rdenom))));
}

#if 0
static std::string dist_t_to_string(dist_t dt) {
    switch(dt) {
        case (dist_t::EUCL):
            return "Euclidean";
        case (dist_t::COS):
            return "Cosine";
        case (dist_t::TAXI):
            return "Taxicab";
        default:
            return "UNKNOWN";
    }
}
#endif

/** \brief Choose the correct distance function and return it
 * \param arg0 A pointer to data
 * \param arg1 Another pointer to data
 * \param len The number of elements used in the comparison
 * \return the distance based on the chosen distance metric
 */
template <typename T>
T dist_comp_raw(const T* arg0, const T* arg1,
        const unsigned len, dist_t dt) {
    switch (dt) {
        case dist_t::EUCL:
            return eucl_dist<T>(arg0, arg1, len);
        case dist_t::COS:
            return cos_dist(arg0, arg1, len);
        case dist_t::TAXI:
            return taxi_dist(arg0, arg1, len);
        case dist_t::SQEUCL:
            return sqeucl_dist<T>(arg0, arg1, len);
        default:
            throw parameter_exception("Unknown distance metric\n");
    }
}

/**
  \brief Used to generate the a stream of random numbers on every processor but
  allow for a parallel and serial impl to generate identical results.
    NOTE: This only works if the data is distributed to processors in the
  same fashion as libElementals <VC, STAR> or <STAR, VC>
**/
template <typename T>
class mpi_random_generator {
private:
    std::uniform_int_distribution<T> _dist;
    std::default_random_engine _gen;
    size_t _nprocs;
    size_t  _rank;
public:
    // End range (end_range) is inclusive i.e random numbers will be
    //      in the inclusive interval (begin_range, end_range)
    mpi_random_generator(const size_t begin_range,
            const size_t end_range, const size_t rank,
            const size_t nprocs, const size_t seed=1234) {
        this->_nprocs = nprocs;
        this->_gen = std::default_random_engine(seed);
        this->_rank = rank;
        this->_dist = std::uniform_int_distribution<T>(begin_range, end_range);
        init();
    }

    void init() {
        for (size_t i = 0; i < _rank; i++)
            _dist(_gen);
    }

    T next() {
        T ret = _dist(_gen);
        for (size_t i = 0; i < _nprocs-1; i++)
            _dist(_gen);
        return ret;
    }
};

/**
  * \brief To avoid further dependencies and because we rarely use this
  *     , we emulate a counter-based random number generator like Random123
  *     to allow for skipping forward in a random stream.
  */
template <typename T>
class rand123emulator {
private:
    std::uniform_int_distribution<T> dist;
    std::default_random_engine gen;

    const void skip(const size_t nskip) {
        for (size_t i = 0; i < nskip; i++)
            dist(gen); // Throw away some random numbers
    }

public:
    rand123emulator(const size_t begin_range, const size_t end_range,
            const size_t nskip, const size_t seed=1234) {
        this->gen = std::default_random_engine(seed);
        this->dist = std::uniform_int_distribution<T>(begin_range, end_range);
        skip(nskip);
    }

    const T next() {
        return dist(gen);
    }
};

template <typename K>
void reset(std::unordered_map<K, unsigned> map) {
    for (auto const& kv : map) {
        map[kv.first] = 0;
    }
}

float time_diff(struct timeval time1, struct timeval time2);
int get_num_omp_threads();
unsigned get_num_nodes();

init_t get_init_type(const std::string init);
dist_t get_dist_type(const std::string dist_type);
void int_handler(int sig_num);
bool is_file_exist(const char *fn);
size_t filesize(const char* filename);

void assert_msg(bool expr, const std::string msg);
} } // End namespace knor::base

namespace knor { namespace test {

template <typename It1, typename It2, typename T>
bool check_collection_equal(It1 arg0first, It1 arg0last,
        It2 arg1first, It2 arg1last, const T tolerance=0,
        const bool check_all=true) {
    size_t pos = 0;
    bool all_eq = true;

    // Are values equal
    while(arg0first != arg0last) {
        T diff = *arg0first - *arg1first;
        if ((T)((diff*diff)/(T)2) > tolerance) {
            std::cerr << "Position " << pos << " mismatch " << *arg0first
                << " !~= " << *arg1first << std::endl;
            all_eq = false;
            if (!check_all)
                return all_eq;
        }
        ++arg0first; ++arg1first;
    }

    // Are lenghts equal
    if (arg1first != arg1last) {
        std::cerr << "Iterator 1 length != Iterator 2 length" << std::endl;
        all_eq = false;
    }
    return all_eq;
}
}} // End namespace knor::test
#endif

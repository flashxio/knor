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

#ifndef __KNOR_TYPES_HPP__
#define __KNOR_TYPES_HPP__

#include <cstddef>
#include <limits>
#include <vector>
#include <string>
#include <utility>
#include "dense_matrix.hpp"

namespace knor {

typedef long long int llong_t;

namespace base {

static const unsigned INVALID_CLUSTER_ID = std::numeric_limits<unsigned>::max();
enum stage_t { INIT, ESTEP }; // What phase of the algo we're in
enum dist_t { EUCL, COS, TAXI, SQEUCL }; // Euclidean, Cosine, Taxicab distance
enum init_t { RANDOM, FORGY, PLUSPLUS, NONE }; // May have to use

class cluster_t {
public:
    size_t nrow, ncol, iters, k;
    std::vector<unsigned> assignments;
    std::vector<size_t> assignment_count;
    std::vector<double> centroids;

    cluster_t(){ }
    cluster_t(const size_t nrow, const size_t ncol, const size_t iters,
             const size_t k, const unsigned* assignments_buf,
             const llong_t* assignment_count_buf,
             const std::vector<double>& centroids);

    cluster_t(const size_t nrow, const size_t ncol, const size_t iters,
             const std::vector<unsigned>& assignments_buf,
             std::vector<llong_t>& assignment_count_buf,
             const std::unordered_map<unsigned, std::vector<double>>& centroids);

    void set_params(const size_t nrow, const size_t ncol, const size_t iters,
            const size_t k) {
        this->nrow = nrow;
        this->ncol = ncol;
        this->iters = iters;
        this->k = k;
    }

    void set_computed(const unsigned* assignments_buf,
            const llong_t* assignment_count_buf,
            const std::vector<double> centroids);

    const void print() const;
    const void write(const std::string dirname) const;
    const bool operator==(const cluster_t& other) const;
    const std::string to_str();

    ~cluster_t() { }
};

struct gmm_t {
    size_t nrow, ncol, iters;
    unsigned k;
    std::vector<double> means;
    std::vector<std::vector<double> > cov_mats;
    std::vector<double> resp_mat;
    std::vector<double> gaussian_prob;

    gmm_t() { }
    bool operator==(const gmm_t& other);
    gmm_t(const size_t nrow, const size_t ncol, const size_t iters,
            const size_t k, double* _means,
            std::vector<base::dense_matrix<double>*>& _cov_mats,
            double* _resp_mat, double* _gaussian_prob);
};

/**
  * A class that stores a map as a vector and is mostly STL compliant
  */
template <typename T>
class vmap_iterator;

template <typename T>
class vmap {
    private:
        std::vector<T> data;
    public:
        T emptyval;
        vmap() : emptyval(0) { }
        vmap(const size_t capacity, const T emptyval) :
                emptyval(emptyval) {
            data.assign(capacity, emptyval);
        }

        void set_capacity(const size_t capacity) {
            data.assign(capacity, emptyval);
        }

        const size_t size() const { return data.size(); }
        void erase(size_t idx) { data[idx] = emptyval; }

        T& operator[] (const size_t idx) {
            if (idx >= size())
                data.resize(idx+1); // TODO: Efficiency
            return data[idx];
        }

        const T& operator[] (const size_t idx) const {
            if (idx >= size())
                throw oob_exception("const vmap::operator[]");
            return data[idx];
        }

        T& at (const size_t idx) {
            if (idx >= size())
                data.resize(idx+1); // TODO: Efficiency
            return (*this)[idx];
        }

        vmap_iterator<T> get_iterator() {
            return vmap_iterator<T>(*this);
        }

        void clear() {
            data.assign(size(), emptyval);
        }

        const bool empty() const { return !static_cast<bool>(size()); }

        const bool has_key(const size_t idx) const {
            if (idx > size()-1)
                return false;
            return (emptyval != data[idx]);
        }

        void get_keys(std::vector<size_t>& ids) {
            for (size_t id = 0; id < size(); id++) {
                if (emptyval != data[id]) {
                    ids.push_back(id);
                }
            }
        }

        const size_t keycount() {
            // TODO: Wasteful, O(capacity)
            size_t count = 0;
            for (size_t i = 0; i < size(); i++) {
                if (emptyval != data[i])
                    count++;
            }
            return count;
        }

        const bool keyless() const {
            // TODO: Wasteful, O(capacity)
            for (auto const& key : data)
                if (emptyval != key)
                    return false;
            return true;
        }

        const void print() const {
#ifndef BIND
            for (size_t i = 0; i < size(); i++) {
                if (emptyval != data[i]) {
                    printf("k: %lu\nv: ", i); data[i]->print_means();
                }
            }
#endif
        }
};

template <typename T>
class vmap_iterator {
    private:
        size_t offset;
        const vmap<T>& vm;

    public:
        vmap_iterator(const vmap<T>& _vm) : vm(_vm) {
            offset = 0;
            for (; offset < vm.size(); offset++) {
                if (vm.emptyval != vm[offset])
                    return;
            }
        }

        // TODO: can we remove copy
        std::pair<size_t, T> next() {
            std::pair<size_t, T> ret(offset, vm[offset]);
            offset++;
            return ret;
        }

        bool has_next() {
            if (offset == vm.size() || !vm.size())
                return false;

            for (; offset < vm.size(); offset++)
                if (vm.emptyval != vm[offset])
                    return true;

            return false;
        }
};
} }
#endif

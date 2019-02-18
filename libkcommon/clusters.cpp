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

#include "io.hpp"
#include "util.hpp"
#include "clusters.hpp"

namespace knor { namespace base {

void clusters::clear() {
    std::fill(means.begin(), means.end(), 0);
    std::fill(num_members_v.begin(), num_members_v.end(), 0);
    std::fill(complete_v.begin(), complete_v.end(), false);
}

/** \param idx the cluster index.
*/
void clusters::set_mean(const kmsvector& mean, const int idx) {
    if (idx == -1) { // Set all means
        means = mean;
    } else {
        std::copy(mean.begin(), mean.end(),
                this->means.begin()+(idx*ncol));
    }
}

void clusters::set_mean(const double* mean, const int idx) {
    if (idx == -1) { // Set all means
        if (means.size() != (ncol*nclust))
            means.resize(ncol*nclust);
        std::copy(&(mean[0]), &(mean[ncol*nclust]), this->means.begin());
    } else {
        std::copy(&(mean[0]), &(mean[ncol]), this->means.begin()+(idx*ncol));
    }
}

void clusters::finalize(const unsigned idx) {
    if (is_complete(idx)) {
        return;
    }

    if (num_members_v[idx] > 1) { // Less than 2 is the same result
        for (unsigned i = 0; i < ncol; i++) {
            means[(idx*ncol)+i] /= double(num_members_v[idx]);
        }
    }
    complete_v[idx] = true;
}

void clusters::unfinalize(const unsigned idx) {
    if (!is_complete(idx))
        return;

    complete_v[idx] = false;
    if (num_members_v[idx] < 2)
        return;

    for (unsigned col = 0; col < ncol; col++) {
        this->means[(ncol*idx) + col] *= (double)num_members_v[idx];
    }
}

void clusters::finalize_all() {
    for (unsigned c = 0;  c < get_nclust(); c++)
        finalize(c);
}

void clusters::unfinalize_all() {
    for (unsigned c = 0;  c < get_nclust(); c++)
        unfinalize(c);
}

void clusters::set_num_members_v(const size_t* arg) {
    std::copy(&(arg[0]), &(arg[nclust]), num_members_v.begin());
}

clusters& clusters::operator=(clusters& other) {
    this->means = other.get_means();
    this->num_members_v = other.get_num_members_v();
    this->ncol = other.get_ncol();
    this->nclust = other.get_nclust();
    return *this;
}

bool clusters::operator==(clusters& other) {
    return (get_ncol() == other.get_ncol() &&
            get_nclust() == other.get_nclust() &&
            v_eq(get_num_members_v(), other.get_num_members_v()) &&
            v_eq(get_means(), other.get_means())
           );
}

clusters& clusters::operator+=(clusters& rhs) {
    for (unsigned i = 0; i < size(); i++)
        this->means[i] += rhs[i];

    for (unsigned idx = 0; idx < nclust; idx++)
        num_members_peq(rhs.get_num_members(idx), idx);
    return *this;
}

void clusters::peq(ptr rhs) {
    assert(rhs->size() == size());
    for (unsigned i = 0; i < size(); i++)
        this->means[i] += rhs->get(i);

    for (unsigned idx = 0; idx < nclust; idx++)
        num_members_peq(rhs->get_num_members(idx), idx);
}

void clusters::means_peq(const double* other) {
    for (unsigned i = 0; i < size(); i++)
        this->means[i] += other[i];
}

void clusters::num_members_v_peq(const size_t* other) {
    for (unsigned i = 0; i < num_members_v.size(); i++)
        this->num_members_v[i] += other[i];
}

// Begin Helpers //
const void clusters::print_means() const {
#ifndef BIND
    printf("nclust: %u\n", get_nclust());
    for (unsigned cl_idx = 0; cl_idx < get_nclust(); cl_idx++) {
        std::cout << "#memb = " << get_num_members(cl_idx) << " ";
        print<double>(&(means[cl_idx*ncol]), ncol);
    }
#endif
}

const void h_clusters::print_means() const {
    clusters::print_means();
#ifndef BIND
    printf("Mean 0 ID: %u\n", zeroid);
    printf("Mean 1 ID: %u\n", oneid);
#endif
}

clusters::clusters(const unsigned nclust, const unsigned ncol) {
    this->nclust = nclust;
    this->ncol = ncol;

    means.resize(ncol*nclust);
    num_members_v.resize(nclust);
    complete_v.assign(nclust, false);
}

clusters::clusters(const unsigned nclust, const unsigned ncol,
        const double* means) {
    this->nclust = nclust;
    this->ncol = ncol;

    set_mean(means);
    num_members_v.resize(nclust);
    complete_v.assign(nclust, true);
}

clusters::clusters(const unsigned nclust, const unsigned ncol,
        const kmsvector& means) : clusters(nclust, ncol, &means[0]) {
}

const void clusters::print_membership_count() const {
    std::string p = "[ ";
    for (unsigned cl_idx = 0; cl_idx < get_nclust(); cl_idx++) {
        p += std::to_string(get_num_members(cl_idx)) + " ";
    }
    p += "]\n";
#ifndef BIND
    std::cout << p;
#endif
}

void clusters::scale_centroid(const double factor,
        const unsigned idx, const double* member) {
    assert(idx < nclust);
    for (unsigned col = 0; col < ncol; col++) {
        means[(ncol*idx)+col] = ((1-factor)*means[(idx*ncol)+col])
            + (factor*(member[col]));
    }
}

// Pruning clusters //
void prune_clusters::reset_s_val_v() {
    std::fill(s_val_v.begin(), s_val_v.end(),
            std::numeric_limits<double>::max());
}

const void prune_clusters::print_prev_means_v() const {
    for (unsigned cl_idx = 0; cl_idx < get_nclust(); cl_idx++) {
        print<double>(&(prev_means[cl_idx*ncol]), ncol);
    }
#ifndef BIND
    std::cout << "\n";
#endif
}

// Sparse clusters
sparse_clusters::sparse_clusters(const unsigned nclust, const unsigned ncol) :
    clusters(nclust, ncol) {
}

/**
  * @param idx: resize to accomodate an index at location idx
  */
void sparse_clusters::resize(const size_t idx) {
    // Quick sanity checks
    assert(nclust == num_members_v.size());
    assert(num_members_v.size() == complete_v.size());
    assert(means.size() == nclust*ncol);


    // These have nclust elements
    auto nelem = (idx+1) - nclust;
    num_members_v.resize(idx+1);
    std::fill_n(&num_members_v[nclust], nelem, 0);

    complete_v.resize(idx+1);
    std::fill_n(complete_v.begin()+nclust, nelem, false);

    auto old_size = means.size();
    nelem = ((idx+1)*ncol) - old_size;

    means.resize((idx+1)*ncol);
    std::fill_n(&means[old_size], nelem, 0); // Zero it out

    nclust = idx + 1;
}

const void sparse_clusters::print_means() const  {
#ifndef BIND
    printf("nclust: %u\n", get_nclust());
    for (size_t i = 0; i < nclust; i++) {
        if (num_members_v[i]) {
            printf("id: %lu\nmean: ", i);
            print(&means[i*ncol], ncol);
        }
    }
#endif
}

void sparse_clusters::peq(ptr rhs) {
    if (rhs->size() > size())
        resize(rhs->size()); // NOTE: should be -1., but ok

    for (unsigned i = 0; i < rhs->size(); i++) // NOTE: rhs could be smaller
        this->means[i] += rhs->get(i);

    for (unsigned idx = 0; idx < rhs->get_nclust(); idx++)
        num_members_peq(rhs->get_num_members(idx), idx);
}

} } // End namespace knor, base

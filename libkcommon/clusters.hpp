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

#ifndef __KNOR_CLUSTERS_HPP__
#define __KNOR_CLUSTERS_HPP__

#include <limits>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>
#include "exception.hpp"
#include "types.hpp"

namespace knor { namespace base {

typedef std::vector<double> kmsvector;
typedef std::vector<double>::iterator kmsiterator;

class clusters {
private:
    double& operator[](const unsigned index) {
        return means[index];
    }

protected:
    // Together are nXd matrix
    unsigned ncol;
    unsigned nclust;
    std::vector<llong_t> num_members_v; // Cluster assignment counts
    std::vector<bool> complete_v; // Have we already divided by num_members

    kmsvector means; // Cluster means

public:
    typedef typename std::shared_ptr<clusters> ptr;

    clusters(const unsigned nclust, const unsigned ncol);
    clusters(const unsigned nclust, const unsigned ncol,
            const kmsvector& means);
    clusters(const unsigned nclust, const unsigned ncol,
            const double* means);

    static ptr create(const unsigned nclust, const unsigned ncol) {
        return ptr(new clusters(nclust, ncol));
    }

    static ptr create(const unsigned nclust, const unsigned ncol,
            const kmsvector& means) {
        return ptr(new clusters(nclust, ncol, means));
    }

    static ptr create(const unsigned nclust, const unsigned ncol,
            const double* means) {
        return ptr(new clusters(nclust, ncol, means));
    }

    const kmsvector& get_means() const {
        return means;
    }

    // Get actual index into mean vector (NOT the cluster ID)
    double get(const unsigned index) { return means[index]; }

    virtual const double* get_mean_rawptr(const size_t idx) const {
        return &means[idx*ncol];
    }

    const llong_t get_num_members(const llong_t idx) const {
        return num_members_v[idx];
    }

    std::vector<llong_t>& get_num_members_v() {
        return num_members_v;
    }

    virtual const bool is_complete(const unsigned idx) {
        return complete_v[idx];
    }

    // NOTE: Thread unsafe
    virtual void set_complete(const unsigned idx, const bool complete=true) {
        complete_v[idx] = complete;
    }

    virtual void set_complete_all(const bool complete=true) {
        for (unsigned c = 0; c < get_nclust(); c++)
            complete_v[c] = complete;
    }

    const unsigned size() const {
        return means.size();
    }

    virtual void num_members_peq(const llong_t val, const unsigned idx) {
        num_members_v[idx] += val;
    }

    // Get an index (based on the entire chunck)
    const unsigned get_ncol() const {
        return ncol;
    }

    const unsigned get_nclust() const {
        return nclust;
    }

    const std::vector<bool>& get_complete_v() const {
        return complete_v;
    }

    virtual void add_member(const double* arr, const unsigned idx) {
        unsigned offset = idx * ncol;
        for (unsigned i=0; i < ncol; i++) {
            means[offset+i] += arr[i];
        }
        num_members_v[idx]++;
    }

    template <typename T>
    void add_member(T& count_it, const unsigned idx) {
        unsigned nid = 0;
        while(count_it.has_next()) {
            double e = count_it.next();
            means[(idx*ncol)+(nid++)] += e;
        }
        num_members_v[idx]++;
    }

    template <typename T>
    void remove_member(T& count_it, const unsigned idx) {
        unsigned nid = 0;
        while(count_it.has_next()) {
            double e = count_it.next();
            means[(idx*ncol)+nid++] -= e;
        }
        num_members_v[idx]--;
    }

    template <typename T>
    void remove_member(const T* arr, const unsigned idx) {
        unsigned offset = idx * ncol;
        for (unsigned i=0; i < ncol; i++) {
            means[offset+i] -= arr[i];
        }
        num_members_v[idx]--;
    }

    template <typename T>
    void swap_membership(const T* arr,
            const unsigned from_idx, const unsigned to_idx) {
        remove_member(arr, from_idx);
        add_member(arr, to_idx);
    }

    template <typename T>
    void swap_membership(T& count_it,
            const unsigned from_id, const unsigned to_id) {
        unsigned nid = 0;
        unsigned from_offset = from_id * ncol;
        unsigned to_offset = to_id * ncol;
        while(count_it.has_next()) {
            double e = count_it.next();
            means[from_offset+nid] -= e;
            means[to_offset+nid++] += e;
        }
        num_members_v[from_id]--;
        num_members_v[to_id]++;
    }

    template<typename T>
    void set_mean(T& it, const int idx) {
        unsigned offset = idx*ncol;
        unsigned nid = 0;
        while(it.has_next()) {
            double e = it.next();
            means[offset+nid] = e;
        }
    }

    clusters& operator+=(clusters& rhs);
    clusters& operator=(clusters& other);
    bool operator==(clusters& other);
    virtual void peq(ptr rhs);
    virtual const void print_means() const;
    virtual void clear();
    /** \param idx the cluster index.
      */
    virtual void set_mean(const kmsvector& mean, const int idx=-1);
    virtual void set_mean(const double* mean, const int idx=-1);
    virtual void finalize(const unsigned idx);
    virtual void unfinalize(const unsigned idx);
    virtual void finalize_all();
    virtual void unfinalize_all();
    virtual void set_num_members_v(const size_t* arg);

    const void print_membership_count() const;
    void means_peq(const double* other);
    virtual void num_members_v_peq(const size_t* other);

    // Used for mini-batch
    void scale_centroid(const double factor,
            const unsigned idx, const double* member);

    virtual void set_zeroid(const unsigned zeroid) { }
    virtual void set_oneid(const unsigned oneid) { }
    virtual void set_id(const unsigned id) { }

    virtual const unsigned get_id() {
        throw abstract_exception();
    }

    virtual const unsigned get_zeroid() {
        throw abstract_exception();
    }

    virtual const unsigned get_oneid() {
        throw abstract_exception();
    }

    virtual void set_converged(const bool status=true) {
        throw abstract_exception();
    }

    virtual const bool has_converged() const {
        throw abstract_exception();
    }

    virtual ~clusters() {}
};

class prune_clusters : public clusters {
private:
    kmsvector s_val_v;
    kmsvector prev_means;
    kmsvector prev_dist_v; // Distance to prev mean

    void init() {
        prev_means.resize(ncol*nclust);
        prev_dist_v.resize(nclust);
        s_val_v.assign(nclust, std::numeric_limits<double>::max());
    }

    prune_clusters(const unsigned nclust, const unsigned ncol):
        clusters(nclust, ncol) {
            init();
    }

    prune_clusters(const unsigned nclust, const unsigned ncol,
            const kmsvector means): clusters(nclust, ncol, means) {
        init();
    }

public:
    typedef typename std::shared_ptr<prune_clusters> ptr;

    static ptr create(const unsigned nclust, const unsigned ncol) {
        return ptr(new prune_clusters(nclust, ncol));
    }

    static ptr create(const unsigned nclust, const unsigned ncol,
            const kmsvector& mean) {
        return ptr(new prune_clusters(nclust, ncol, mean));
    }

    void set_s_val(const double val, const unsigned idx) {
        s_val_v[idx] = val;
    }

    double const get_s_val(const unsigned idx) { return s_val_v[idx]; }

    const kmsvector& get_prev_means() const {
        return prev_means;
    }

    void set_prev_means() {
        this->prev_means = means;
    }

    void set_prev_dist(const double dist, const unsigned idx) {
        prev_dist_v[idx] = dist;
    }

    double get_prev_dist(const unsigned idx) {
        return prev_dist_v[idx];
    }

    const void print_prev_means_v() const;
    void reset_s_val_v();
};

class h_clusters : public clusters {
private:
    // This cluster's ID and that of 0 and 1
    unsigned id, zeroid, oneid;
    bool converged;
public:
    using clusters::clusters;
    std::vector<double> metadata; // Wildcard data for the clusters

    static ptr create(const unsigned nclust, const unsigned ncol) {
        auto ret = ptr(new h_clusters(nclust, ncol));
        ret->set_converged(false);
        return ret;
    }

    static ptr create(const unsigned nclust, const unsigned ncol,
            const unsigned id, const unsigned zeroid, const unsigned oneid) {
        auto ret = ptr(new h_clusters(nclust, ncol));
        ret->set_converged(false);
        ret->set_id(id);
        ret->set_zeroid(zeroid);
        ret->set_oneid(oneid);
        return ret;
    }

    static ptr create(const unsigned nclust, const unsigned ncol,
            const double* centers) {
        auto ret = ptr(new h_clusters(nclust, ncol, centers));
        ret->set_converged(false);
        return ret;
    }

    static std::shared_ptr<h_clusters> cast2(ptr obj) {
        return std::static_pointer_cast<h_clusters>(obj);
    }

    void set_zeroid(const unsigned zeroid) override  {
        this->zeroid = zeroid;
    }

    void set_oneid(const unsigned oneid) override {
        this->oneid = oneid;
    }

    const unsigned get_zeroid() override {
        return zeroid;
    }

    const unsigned get_oneid() override {
        return oneid;
    }

    void set_id(const unsigned id) override {
        this->id = id;
    }

    const unsigned get_id() override {
        return id;
    }

    void set_converged(const bool status) override {
        converged = status;
    }

    const bool has_converged() const override {
        return this->converged;
    }

    const void print_means() const override;
};

// Begin sparse_clusters
class sparse_clusters : public clusters {
    private:
        //std::vector<size_t> index; // TODO: Actually sparsify Data index
        void resize(const size_t idx);

    public:

    typedef typename std::shared_ptr<clusters> ptr;
    sparse_clusters(const unsigned nclust, const unsigned ncol);

    static ptr create(const unsigned nclust, const unsigned ncol) {
        return ptr(new sparse_clusters(nclust, ncol));
    }

    void num_members_peq(const llong_t val, const unsigned idx) override {
        if (idx >= nclust)
            throw oob_exception("sparse_clusters::num_members_peq");
        clusters::num_members_peq(val, idx);
    }

    const bool is_complete(const unsigned idx) override {
        if (idx >= nclust)
            throw oob_exception("sparse_clusters::is_complete");
        return clusters::is_complete(idx);
    }

    void set_complete(const unsigned idx, const bool complete=true) override {
        if (idx >= nclust)
            resize(idx);
        clusters::set_complete(idx, complete);
    }

    const double* get_mean_rawptr(const size_t idx) const override {
        if (idx >= nclust)
            throw oob_exception("get_mean_rawptr::get_mean_rawptr");

        return clusters::get_mean_rawptr(idx);
    }


    void add_member(const double* arr, const unsigned idx) override {
        if (idx >= nclust)
            resize(idx); // +1 for 0-based indexing

        clusters::add_member(arr, idx);
    }

    const void print_means() const override;

    void set_mean(const kmsvector& mean, const int idx=-1) override {
        throw abstract_exception();
    }
    void set_mean(const double* mean, const int idx=-1) override {
        throw abstract_exception();
    }

    void peq(ptr rhs) override;
};

} } // End namespace knor, base
#endif

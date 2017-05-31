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

#ifndef __KPM_BASE_KMEANS_COORDINATOR_HPP__
#define __KPM_BASE_KMEANS_COORDINATOR_HPP__

#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>

#include "kmeans_types.hpp"
#include "thread_state.hpp"
#include "exception.hpp"

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

namespace kpmeans {

class base_kmeans_thread;

class base_kmeans_coordinator {
protected:
    unsigned nthreads, nnodes;
    size_t nrow, ncol;
    std::string fn; // file on disk
    unsigned* cluster_assignments;
    size_t* cluster_assignment_counts;
    unsigned k;
    kpmeans::base::init_type_t _init_t;
    kpmeans::base::dist_type_t _dist_t;
    double tolerance;
    unsigned max_iters;
    size_t num_changed; // total # samples changed in an iter
    // how many threads have not completed their task
    std::atomic<unsigned> pending_threads;

    // threading
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_mutexattr_t mutex_attr;
    std::vector<std::shared_ptr<base_kmeans_thread> > threads;

    base_kmeans_coordinator(const std::string fn, const size_t nrow,
            const size_t ncol, const unsigned k, const unsigned max_iters,
            const unsigned nnodes, const unsigned nthreads,
            const double* centers, const kpmeans::base::init_type_t it,
            const double tolerance, const kpmeans::base::dist_type_t dt);

public:
    const size_t get_num_changed() const { return num_changed; }
    typedef std::shared_ptr<base_kmeans_coordinator> ptr;
    typedef std::vector<std::shared_ptr
        <base_kmeans_thread> >::iterator thread_iter;

    // pass file handle to threads to read & numa alloc
    virtual void run_init() = 0;
    virtual void random_partition_init() = 0;
    virtual void forgy_init() = 0;

    virtual kpmeans::base::kmeans_t run_kmeans(
            double* allocd_data=NULL, const bool numa_opt=false) = 0;
    virtual void kmeanspp_init() = 0;
    virtual void wake4run(kpmeans::thread_state_t state) = 0;
    virtual const double* get_thd_data(const unsigned row_id) const = 0;

    virtual void set_thread_clust_idx(const unsigned clust_idx) = 0;
    virtual double reduction_on_cuml_sum() = 0;
    virtual void destroy_threads() = 0;
    virtual void set_thd_dist_v_ptr(double* v) = 0;
    void wait4complete();
    std::vector<std::shared_ptr<base_kmeans_thread> >& get_threads() {
        return threads;
    }

    virtual void set_thread_data_ptr(double* allocd_data);

    virtual void set_global_ptrs() { throw kpmeans::base::abstract_exception(); };
    virtual const void print_thread_data() {
        throw kpmeans::base::abstract_exception();
    };
    virtual void build_thread_state() { throw kpmeans::base::abstract_exception(); };
    const unsigned* get_cluster_assignments() const {
        return cluster_assignments;
    }

    void clear_cluster_assignments() {
        std::fill(cluster_assignments,
                cluster_assignments+nrow, kpmeans::base::INVALID_CLUSTER_ID);
    }
    const size_t get_nrow() { return nrow; }
    const size_t get_ncol() { return ncol; }
};
} // namespace kpmeans
#endif

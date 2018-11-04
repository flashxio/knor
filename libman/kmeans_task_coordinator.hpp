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
#ifndef __KNOR_KMEANS_TASK_COORDINATOR_HPP__
#define __KNOR_KMEANS_TASK_COORDINATOR_HPP__

#include "coordinator.hpp"
#include "util.hpp"

namespace knor {
class task;

namespace base {
    class prune_clusters;
    class thd_safe_bool_vector;
}

namespace prune {
class kmeans_task_thread;
class dist_matrix;

class kmeans_task_coordinator : public knor::coordinator {
protected: // So lazy ..
    // Metadata
    // max index stored within each threads partition
    std::shared_ptr<base::prune_clusters> cltrs;
    std::shared_ptr<base::thd_safe_bool_vector> recalculated_v;
    std::vector<double> dist_v; // global
    std::shared_ptr<dist_matrix> dm;

    // For kmeansPP
    std::default_random_engine generator;
    std::uniform_real_distribution<double> ur_distribution;
    std::uniform_int_distribution<unsigned> ui_distribution;
    bool inited;

    // For mini-batching
    unsigned mb_size;

    kmeans_task_coordinator(const std::string fn, const size_t nrow,
            const size_t ncol, const unsigned k, const unsigned max_iters,
            const unsigned nnodes, const unsigned nthreads,
            const double* centers, const base::init_t it,
            const double tolerance, const base::dist_t dt);

public:
    static coordinator::ptr create(
            const std::string fn, const size_t nrow,
            const size_t ncol, const unsigned k, const unsigned max_iters,
            const unsigned nnodes, const unsigned nthreads,
            const double* centers=NULL, const std::string init="kmeanspp",
            const double tolerance=-1, const std::string dist_type="eucl") {

        base::init_t _init_t = base::get_init_type(init);
        base::dist_t _dist_t = base::get_dist_type(dist_type);

#if KM_TEST
#ifndef BIND
        printf("kmeans task coordinator => NUMA nodes: %u, nthreads: %u, "
                "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                dist_type.c_str(), fn.c_str());
#endif
#endif
        return coordinator::ptr(
                new kmeans_task_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t, tolerance, _dist_t));
    }

    std::shared_ptr<base::prune_clusters> get_gcltrs() {
        return cltrs;
    }

    // Mini-batch
    void set_mini_batch_size(const unsigned mb_size) { this->mb_size = mb_size; }
    const unsigned get_mini_batch_size() { return mb_size; }
    void mb_iteration_end();
    // End Mini-batch

    std::shared_ptr<dist_matrix> get_dm() {
        return dm;
    }

    // For standalone kmeansPP
    double compute_cluster_energy();
    void reinit();
    base::cluster_t dump_state();
    void tally_assignment_counts();

    // Pass file handle to threads to read & numa alloc
    void update_clusters(const bool prune_init);
    void set_global_ptrs() override;
    void set_thread_data_ptr(double* allocd_data) override;
    virtual void kmeanspp_init() override;
    virtual void random_partition_init() override;
    virtual void forgy_init() override;
    virtual base::cluster_t run(double* allocd_data=NULL,
            const bool numa_opt=false) override;

    base::cluster_t mb_run(double* allocd_data=NULL);

    void set_task_data_ptrs();

    void set_prune_init(const bool prune_init);
    virtual void build_thread_state() override;
};
} } // End namespace knor, prune
#endif

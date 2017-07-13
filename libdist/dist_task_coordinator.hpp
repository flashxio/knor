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
#ifndef __KPM_DIST_TASK_COORDINATOR_HPP__
#define __KPM_DIST_TASK_COORDINATOR_HPP__

#include "kmeans_task_coordinator.hpp"
#include "exception.hpp"

namespace kpmbase = kpmeans::base;

namespace kpmeans { namespace prune {

constexpr unsigned root = 0;

class dist_task_coordinator : public kmeans_task_coordinator {
private:
    dist_task_coordinator(int argc, char* argv[],
            const std::string fn, const size_t nrow,
            const size_t ncol, const unsigned k, const unsigned max_iters,
            const unsigned nnodes, const unsigned nthreads,
            const double* centers, const kpmbase::init_type_t it,
            const double tolerance, const kpmbase::dist_type_t dt);

    int mpi_rank;
    int nprocs;
    size_t g_nrow;
    std::vector<size_t> prev_num_members;

public:
    static base_kmeans_coordinator::ptr create(int argc, char* argv[],
            const std::string fn, const size_t nrow,
            const size_t ncol, const unsigned k, const unsigned max_iters,
            const unsigned nnodes, const unsigned nthreads,
            const double* centers=NULL, const std::string init="kmeanspp",
            const double tolerance=-1, const std::string dist_type="eucl") {

        kpmbase::init_type_t _init_t = kpmbase::get_init_type(init);
        kpmbase::dist_type_t _dist_t = kpmbase::get_dist_type(dist_type);

#if KM_TEST
        printf("kmeans task coordinator => NUMA nodes: %u, nthreads: %u, "
                "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                dist_type.c_str(), fn.c_str());
#endif
        return base_kmeans_coordinator::ptr(
                new dist_task_coordinator(argc, argv, fn, nrow, ncol, k,
                    max_iters, nnodes, nthreads, centers,
                    _init_t, tolerance, _dist_t));
    }

    const void print_thread_data() override;

    // Must override routines
    void kmeanspp_init() override;
    void random_partition_init() override;
    void forgy_init() override;
    void run_kmeans(kpmbase::kmeans_t& ret, const std::string outdir="");

    const bool is_local(const size_t global_rid) const;
    const size_t global_rid(const size_t local_rid) const;
    const size_t local_rid(const size_t global_rid) const;
    void pp_aggregate();
    std::vector<size_t>& get_prev_num_members() {
        return prev_num_members;
    }
    const int get_nprocs() const { return nprocs; }
    const size_t init(int argc, char* argv[], const size_t g_nrow);
    ~dist_task_coordinator();
};
} } // End namespace kpmeans, prune
#endif

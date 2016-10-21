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

namespace kpmeans { namespace prune {

class dist_task_coordinator : public kpmprune::kmeans_task_coordinator {
private:
    dist_task_coordinator(const std::string fn, const size_t nrow,
            const size_t ncol, const unsigned k, const unsigned max_iters,
            const unsigned nnodes, const unsigned nthreads,
            const unsigned mpi_rank, const unsigned nprocs,
            const double* centers, const kpmbase::init_type_t it,
            const double tolerance, const kpmbase::dist_type_t dt);

    unsigned mpi_rank;
    unsigned nprocs;
    size_t start_rid_offset; // Per process offset into the file

    /* NOTE
        nrow: The number of rows LOCAL to the process
    */
    size_t const get_proc_rows(const size_t g_nrow,
            const unsigned nprocs, const unsigned mpi_rank) const;

public:
    static base_kmeans_coordinator::ptr create(
            const std::string fn, const size_t nrow,
            const size_t ncol, const unsigned k, const unsigned max_iters,
            const unsigned nnodes, const unsigned nthreads,
            const unsigned mpi_rank, const unsigned nprocs,
            const double* centers=NULL, const std::string init="kmeanspp",
            const double tolerance=-1, const std::string dist_type="eucl") {

        kpmbase::init_type_t _init_t = kpmbase::get_init_type(init);
        if (_init_t != kpmbase::init_type_t::RANDOM)
            throw kpmbase::not_implemented_exception();

        kpmbase::dist_type_t _dist_t = kpmbase::get_dist_type(dist_type);

#if KM_TEST
        printf("kmeans task coordinator => NUMA nodes: %u, nthreads: %u, "
                "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                dist_type.c_str(), fn.c_str());
#endif
        return base_kmeans_coordinator::ptr(
                new dist_task_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, mpi_rank, nprocs, centers,
                    _init_t, tolerance, _dist_t));
    }

    const void print_thread_data();
    std::pair<size_t, size_t> get_rid_len_tup(const unsigned thd_id);

    // Must override routines
    void kmeanspp_init();
    void random_partition_init();
    void forgy_init();
    void run_kmeans(); /*Run a single iteration*/
};
} } // End namespace kpmeans, prune
#endif

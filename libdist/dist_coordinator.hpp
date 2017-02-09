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
#ifndef __KPM_DIST_COORDINATOR_HPP__
#define __KPM_DIST_COORDINATOR_HPP__

#include "kmeans_coordinator.hpp"
#include "exception.hpp"

namespace kpmeans { namespace dist {

class dist_coordinator : public kpmeans::kmeans_coordinator {
private:
    dist_coordinator(const std::string fn, const size_t nrow,
            const size_t ncol, const unsigned k, const unsigned max_iters,
            const unsigned nnodes, const unsigned nthreads,
            const unsigned mpi_rank, const unsigned nprocs,
            const double* centers, const kpmbase::init_type_t it,
            const double tolerance, const kpmbase::dist_type_t dt);

    unsigned mpi_rank;
    unsigned nprocs;
    size_t g_nrow;

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
        kpmbase::dist_type_t _dist_t = kpmbase::get_dist_type(dist_type);

        return base_kmeans_coordinator::ptr(
                new dist_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, mpi_rank, nprocs, centers,
                    _init_t, tolerance, _dist_t));
    }

    const void print_thread_data() override;

    // Must override routines
    void kmeanspp_init() override;
    void random_partition_init() override;
    void forgy_init() override;
    const bool is_local(const size_t global_rid) const;
    kpmbase::kmeans_t run_kmeans() override; /*Run a single iteration*/

    const size_t global_rid(const size_t local_rid) const;
    const size_t local_rid(const size_t global_rid) const;
    void pp_aggregate();
    void shift_thread_start_rid();
};
} } // End namespace kpmeans::dist
#endif

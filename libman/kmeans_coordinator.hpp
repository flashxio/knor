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
#ifndef __KPM_KMEANS_COORDINATOR_HPP__
#define __KPM_KMEANS_COORDINATOR_HPP__

#include "base_kmeans_coordinator.hpp"
#include "util.hpp"

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

namespace kpmbase = kpmeans::base;
namespace kpmeans {
class base_kmeans_thread;
    namespace base {
    class clusters;
} }

namespace kpmeans {
class kmeans_coordinator : public kpmeans::base_kmeans_coordinator {
    protected:
        // Metadata
        // max index stored within each threads partition
        std::vector<unsigned> thd_max_row_idx;
        std::shared_ptr<kpmbase::clusters> cltrs;

        kmeans_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const kpmbase::init_type_t it,
                const double tolerance, const kpmbase::dist_type_t dt);

    public:
        static base_kmeans_coordinator::ptr create(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers=NULL, const std::string init="kmeanspp",
                const double tolerance=-1, const std::string dist_type="eucl") {

            kpmbase::init_type_t _init_t = kpmbase::get_init_type(init);
            kpmbase::dist_type_t _dist_t = kpmbase::get_dist_type(dist_type);
#if KM_TEST
            printf("kmeans coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                    dist_type.c_str(), fn.c_str());
#endif
            return base_kmeans_coordinator::ptr(
                    new kmeans_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t, tolerance, _dist_t));
        }

        std::shared_ptr<kpmbase::clusters> get_gcltrs() {
            return cltrs;
        }

        std::pair<unsigned, unsigned> get_rid_len_tup(const unsigned thd_id);
        // Pass file handle to threads to read & numa alloc
        virtual kpmbase::kmeans_t run_kmeans(double* allocd_data,
                const bool numa_opt) override;
        void update_clusters();
        void kmeanspp_init() override;
        void wake4run(kpmeans::thread_state_t state) override;
        void destroy_threads() override;
        void set_thread_clust_idx(const unsigned clust_idx) override;
        double reduction_on_cuml_sum() override;
        void set_thd_dist_v_ptr(double* v) override;
        void run_init() override;
        void random_partition_init() override;
        void forgy_init() override;
        const double* get_thd_data(const unsigned row_id) const override;
        ~kmeans_coordinator();

        // For testing
        void const print_thread_data() override;
        void build_thread_state() override;
        void const print_thread_start_rids();
};
}
#endif

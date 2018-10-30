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
#ifndef __KNOR_FCM_COORDINATOR_HPP__
#define __KNOR_FCM_COORDINATOR_HPP__

#include "coordinator.hpp"
#include "util.hpp"

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

namespace knor {

class fcm_coordinator : public coordinator {
    protected:
        // Metadata
        // max index stored within each threads partition
        std::vector<unsigned> thd_max_row_idx;
        base::dense_matrix<double>* centers; // k x ncol
        base::dense_matrix<double>* prev_centers; // k x ncol
        base::dense_matrix<double>* um; // nrow x k
        unsigned fuzzindex;

        fcm_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const base::init_t it,
                const double tolerance, const base::dist_t dt,
                const unsigned fuzzindex);

    public:
        static coordinator::ptr create(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers=NULL, const std::string init="forgy",
                const double tolerance=-1, const std::string dist_type="cos",
                const unsigned fuzzindex=2) {

            base::init_t _init_t = base::get_init_type(init);
            base::dist_t _dist_t = base::get_dist_type(dist_type);
#if KM_TEST
#ifndef BIND
            printf("fcm_coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                    dist_type.c_str(), fn.c_str());
#endif
#endif
            return coordinator::ptr(
                    new fcm_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t,
                    tolerance, _dist_t, fuzzindex));
        }

        std::pair<unsigned, unsigned> get_rid_len_tup(const unsigned thd_id);
        // Pass file handle to threads to read & numa alloc
        base::cluster_t run(double* allocd_data,
                const bool numa_opt) override {
            throw base::not_implemented_exception();
        }

        void kmeanspp_init();
        base::cluster_t soft_run(double* allocd_data=NULL);
        void update_contribution_matrix();
        void update_centers();
        void wake4run(knor::thread_state_t state) override;
        void destroy_threads() override;
        void set_thread_clust_idx(const unsigned clust_idx) override;
        double reduction_on_cuml_sum() override;
        void set_thd_dist_v_ptr(double* v) override;
        void run_init() override;
        void random_partition_init() override {
            throw knor::base::abstract_exception(); };
        void forgy_init() override;
        const double* get_thd_data(const unsigned row_id) const override;
        ~fcm_coordinator();

        // For testing
        void const print_thread_data() override;
        virtual void build_thread_state() override;
        void const print_thread_start_rids();
};
}
#endif

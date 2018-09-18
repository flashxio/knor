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
#ifndef __KNOR_PAM_COORDINATOR_HPP__
#define __KNOR_PAM_COORDINATOR_HPP__

#include "coordinator.hpp"
#include "util.hpp"

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

namespace knor {

namespace base {
    class clusters;
}
namespace prune {
    class dist_matrix;
}

class thread;

class pam_coordinator : public coordinator {
    protected:
        // Metadata
        // max index stored within each threads partition
        std::vector<unsigned> thd_max_row_idx;
        std::shared_ptr<base::clusters> cltrs;

        // Pairwise distances for all samples
        std::shared_ptr<prune::dist_matrix> pw_dm;

        pam_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const base::init_t it,
                const double tolerance, const base::dist_t dt);

    public:
        static coordinator::ptr create(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers=NULL, const std::string init="random",
                const double tolerance=-1, const std::string dist_type="taxi") {

            base::init_t _init_t = base::get_init_type(init);
            base::dist_t _dist_t = base::get_dist_type(dist_type);
#if KM_TEST
#ifndef BIND
            printf("PAM coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                    dist_type.c_str(), fn.c_str());
#endif
#endif
            return coordinator::ptr(
                    new pam_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t, tolerance, _dist_t));
        }

        std::shared_ptr<base::clusters> get_gcltrs() {
            return cltrs;
        }

        std::pair<unsigned, unsigned> get_rid_len_tup(const unsigned thd_id);
        // Pass file handle to threads to read & numa alloc
        virtual base::kmeans_t run(double* allocd_data,
                const bool numa_opt) override;
        void update_clusters();
        void wake4run(knor::thread_state_t state) override;
        void destroy_threads() override;
        void set_thread_clust_idx(const unsigned clust_idx) override;
        double reduction_on_cuml_sum() override;
        void run_init() override;
        void random_partition_init() override;
        const double* get_thd_data(const unsigned row_id) const override;
        ~pam_coordinator();

        // For testing
        void const print_thread_data() override;
        void build_thread_state() override;
        void const print_thread_start_rids();
};
}
#endif
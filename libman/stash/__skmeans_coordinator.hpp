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
#ifndef __KNOR_SKMEANS_COORDINATOR_HPP__
#define __KNOR_SKMEANS_COORDINATOR_HPP__

#include "kmeans_coordinator.hpp"
#include "types.hpp"

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

namespace knor {

    namespace base {
        class cluster_t;
    }

class thread;

class skmeans_coordinator : public kmeans_coordinator {
    private:
        void bounds_reduction();
        std::vector<double> g_feature_max;
        std::vector<double> g_feature_min;

        skmeans_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const base::init_t it,
                const double tolerance, const base::dist_t dt);

    public:

        static coordinator::ptr create(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers=NULL, const std::string init="kmeanspp",
                const double tolerance=-1, const std::string dist_type="taxi") {

            base::init_t _init_t = base::get_init_type(init);
            base::dist_t _dist_t = base::get_dist_type(dist_type);

            printf("skmeans coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                    dist_type.c_str(), fn.c_str());

            return coordinator::ptr(
                    new skmeans_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t, tolerance, _dist_t));
        }

        void build_thread_state() override;
        void preprocess_data() override;
        void test() { printf("skmeans_coordinator::test()\n"); }

        base::cluster_t run(double* allocd_data,
                const bool numa_opt=false) override;
};
}
#endif
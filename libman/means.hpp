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
#ifndef __KNOR_MEANS_HPP__
#define __KNOR_MEANS_HPP__

#include "base.hpp"
#include "util.hpp"

namespace knor {

namespace core {
    class clusters;
}

class means : public base {
    protected:
        // Metadata
        // max index stored within each threads partition
        std::shared_ptr<core::clusters> cltrs;

        means(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const core::init_t it,
                const double tolerance, const core::dist_t dt);

    public:
        static base::ptr create(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers=NULL, const std::string init="kmeanspp",
                const double tolerance=-1, const std::string dist_type="eucl") {

            core::init_t _init_t = core::get_init_type(init);
            core::dist_t _dist_t = core::get_dist_type(dist_type);
#if KM_TEST
#ifndef BIND
            printf("kmeans coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                    dist_type.c_str(), fn.c_str());
#endif
#endif
            return base::ptr(
                    new means(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t, tolerance, _dist_t));
        }

        std::shared_ptr<core::clusters> get_gcltrs() {
            return cltrs;
        }

        // Pass file handle to threads to read & numa alloc
        virtual core::cluster_t run(double* allocd_data=NULL,
            const bool numa_opt=false) override;
        void update_clusters();
        void kmeanspp_init() override;
        void random_partition_init() override;
        void forgy_init() override;
        virtual void preprocess_data() {
            throw knor::core::abstract_exception();
        }
        virtual void build_thread_state() override;
};
}
#endif
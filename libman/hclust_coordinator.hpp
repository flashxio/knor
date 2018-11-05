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
#ifndef __KNOR_HCLUST_COORDINATOR_HPP__
#define __KNOR_HCLUST_COORDINATOR_HPP__

#include <mutex>
#include <unordered_map>

#include "coordinator.hpp"
#include "util.hpp"

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

namespace knor {

namespace base {
    class clusters;
}

class id_generator;
static std::shared_ptr<id_generator> ider = nullptr;

// Singleton class
class id_generator {
    private:
        unsigned current_id;
        std::mutex _mutex;

        id_generator() {
            current_id = 0;
        };


    public:
        typedef std::shared_ptr<id_generator> ptr;

        static ptr get_generator() {
            if (nullptr == ider)
                ider = ptr(new id_generator());
            return ider;
        };

        unsigned next() {
            _mutex.lock();
            unsigned ret = current_id++;
            _mutex.unlock();
            return ret;
        }
};

class hclust_coordinator : public coordinator {
    protected:
        std::unordered_map<unsigned, std::shared_ptr<base::clusters>>* hcltrs;

        hclust_coordinator(const std::string fn, const size_t nrow,
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
                const double tolerance=-1, const std::string dist_type="eucl") {

            base::init_t _init_t = base::get_init_type(init);
            base::dist_t _dist_t = base::get_dist_type(dist_type);
#if KM_TEST
#ifndef BIND
            printf("hclust coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                    dist_type.c_str(), fn.c_str());
#endif
#endif
            return coordinator::ptr(
                    new hclust_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t, tolerance, _dist_t));
        }

        // Pass file handle to threads to read & numa alloc
        virtual base::cluster_t run(double* allocd_data=NULL,
            const bool numa_opt=false) override;
        //void update_clusters();
        void kmeanspp_init() override;
        void random_partition_init() override;
        void forgy_init() override;
        virtual void preprocess_data() {
            throw knor::base::abstract_exception();
        }
        virtual void build_thread_state() override;
        ~hclust_coordinator();
};
}
#endif

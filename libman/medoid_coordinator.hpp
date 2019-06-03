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
#ifndef __KNOR_MEDOID_COORDINATOR_HPP__
#define __KNOR_MEDOID_COORDINATOR_HPP__

#include "coordinator.hpp"
#include "util.hpp"

namespace knor {

namespace base {
    class clusters;
}

class thread;

class medoid_coordinator : public coordinator {
    protected:
        // Metadata
        // max index stored within each threads partition
        std::shared_ptr<base::clusters> cltrs;
        // Cumulative distance of all members from the mediod
        std::vector<double> medoid_energy;
        bool medoids_changed;
        double sample_rate;

        // Need for medoid step. Contains the rid of members of each cluster
        std::vector<std::vector<unsigned> > membership;

        medoid_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const base::init_t it,
                const double tolerance, const base::dist_t dt,
                const double sample_rate);

    public:
        static coordinator::ptr create(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers=NULL, const std::string init="forgy",
                const double tolerance=-1, const std::string dist_type="taxi",
                const double sample_rate=.2) {

            base::init_t _init_t = base::get_init_type(init);
            base::dist_t _dist_t = base::get_dist_type(dist_type);
#if KM_TEST
#ifndef BIND
            printf("medoid coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                    dist_type.c_str(), fn.c_str());
#endif
#endif
            return coordinator::ptr(
                    new medoid_coordinator(fn, nrow, ncol, k, max_iters,
                    nnodes, nthreads, centers, _init_t, tolerance, _dist_t,
                    sample_rate));
        }

        std::shared_ptr<base::clusters> get_gcltrs() {
            return cltrs;
        }

        void clear_membership() {
            for (auto& v : membership)
                v.clear();
        }

        std::vector<std::vector<unsigned> >& get_membership() {
            return membership;
        }

        // Pass file handle to threads to read & numa alloc
        virtual base::cluster_t run(double* allocd_data,
                const bool numa_opt=false) override;
        void update_clusters();
        void run_init() override;
        void forgy_init() override;
        void build_thread_state() override;

        // medoid specific
        void populate_membership();
        void compute_globals();
        void sanity_check(); // Always call compute_globals before this
        void choose_global_medoids(double* gdata=NULL);
};
}
#endif

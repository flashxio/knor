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
#include "types.hpp"

#include "coordinator.hpp"
#include "util.hpp"

namespace knor {

namespace base {
    class clusters;
    class h_clusters;
    class thd_safe_bool_vector;
}

class hclust_id_generator;

struct c_part {
    unsigned l0;
    unsigned l1;
    unsigned r0;
    unsigned r1;

    c_part() {
        l0 = l1 = r0 = r1 = std::numeric_limits<unsigned>::max();
    }

    bool l_splittable() const {
        return (l0 != std::numeric_limits<unsigned>::max() &&
                l1 != std::numeric_limits<unsigned>::max());
    }

    bool r_splittable() const {
        return (r0 != std::numeric_limits<unsigned>::max() &&
                r1 != std::numeric_limits<unsigned>::max());
    }

    void print() const {
#ifndef BIND
        printf("l0: %u, l1: %u, r0: %u, r1: %u\n",
            l0, l1, r0, r1);
#endif
    }
    bool splittable() const { return l_splittable() && r_splittable(); }
    void check() const { assert(splittable()); }
};

class hclust_coordinator : public coordinator {
    protected:
        base::vmap<std::shared_ptr<base::clusters>> hcltrs;
        size_t max_nodes;

        base::vmap<unsigned> nchanged;
        // Whether a particular cluster is cluster is still actively splitting
        //  Multithreaded write
        std::shared_ptr<base::thd_safe_bool_vector> cltr_active_vec;
        std::default_random_engine ui_generator;
        std::uniform_int_distribution<unsigned> ui_distribution;
        std::mutex _mutex;
        // Keep track of the parent cluster id (partition id)
        std::vector<unsigned> part_id;
        std::shared_ptr<hclust_id_generator> ider;
        unsigned min_clust_size;
        std::unordered_map<unsigned, std::vector<double>> final_centroids;
        size_t curr_nclust;
    public:
        hclust_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers, const base::init_t it,
                const double tolerance, const base::dist_t dt,
                const unsigned min_clust_size);

        static coordinator::ptr create(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nnodes, const unsigned nthreads,
                const double* centers=NULL, const std::string init="forgy",
                const double tolerance=-1, const std::string dist_type="eucl",
                const unsigned min_clust_size=2) {

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
                    nnodes, nthreads, centers, _init_t, tolerance,
                    _dist_t, min_clust_size));
        }

        virtual void run_hinit();
        virtual unsigned forgy_select(const unsigned cid);
        virtual void print_clusters();

        // Pass file handle to threads to read & numa alloc
        virtual base::cluster_t run(double* allocd_data=NULL,
            const bool numa_opt=false) override;
        virtual void update_clusters();
        virtual void forgy_init() override;
        void none_init();
        virtual void preprocess_data() {
            throw knor::base::abstract_exception();
        }

        // Used to take the mean of the full dataset
        virtual void build_thread_state() override;
        virtual void partition_mean(base::vmap<
        std::shared_ptr<base::clusters>>& part_hcltrs);
        virtual void init_splits();
        virtual void inner_init(std::vector<unsigned>& remove_cache);
        virtual void spawn(const unsigned& zeroid,
                const unsigned& oneid, const c_part& cp);
        virtual void deactivate(const unsigned id);
        virtual void activate(const unsigned id);
        virtual const bool is_active(const unsigned i) const ;
        virtual const bool steady_state() const;
        virtual void accumulate_cluster_counts();
        virtual void complete_final_centroids();
        virtual bool at_cluster_cap() { return curr_nclust > k*2; }
        void verify_consistency();
};
}
#endif

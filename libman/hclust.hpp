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

#ifndef __KNOR_HCLUST_HPP__
#define __KNOR_HCLUST_HPP__

#include "types.hpp"
#include "thread.hpp"

namespace knor {
namespace base {
    class clusters;
    class h_clusters;
    class thd_safe_bool_vector;
}

typedef base::vmap<std::shared_ptr<base::clusters>> hclust_map;

class hclust : public thread {
    protected:
         // Pointer to global cluster data
        hclust_map& g_hcltrs;
        hclust_map local_hcltrs;
        base::vmap<unsigned> nchanged; // How many change in each partition
        const std::shared_ptr<base::thd_safe_bool_vector> cltr_active_vec; // Clusters still active

        unsigned k;
        unsigned nprocrows; // The number of rows in this threads partition
        unsigned* part_id;

        hclust(const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol, unsigned k,
                hclust_map& g_hcltrs,
                unsigned* cluster_assignments,
                const std::string fn, base::dist_t dist_metric,
                const std::shared_ptr<base::thd_safe_bool_vector> cltr_active_vec);
    public:
        static thread::ptr create(
                const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol, unsigned k,
                hclust_map& g_hcltrs,
                unsigned* cluster_assignments, const std::string fn,
                base::dist_t dist_metric,
                const std::shared_ptr<base::thd_safe_bool_vector> cltr_active_vec) {
            return thread::ptr(
                        new hclust(node_id, thd_id, start_rid,
                        nprocrows, ncol, k, g_hcltrs,
                        cluster_assignments, fn, dist_metric, cltr_active_vec));
        }


        virtual void set_part_id(unsigned* part_id) {
            this->part_id = part_id;
        }

        const base::vmap<unsigned>& get_nchanged() const {
            return this->nchanged;
        }

        hclust_map& get_local_hcltrs() {
            return local_hcltrs;
        }

        virtual void start(const thread_state_t state) override;
        // Given the current ID split it into two (or not)
        virtual void H_split_step() { };
        virtual void H_EM_step(); // Similar to EM step
        virtual void partition_mean() { };
        virtual void run() override;
};
} // End namespace knor
#endif

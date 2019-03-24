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

#ifndef __KNOR_MEDOID_HPP__
#define __KNOR_MEDOID_HPP__

#include <vector>
#include "thread.hpp"
#include <random>

namespace kprune = knor::prune;

namespace knor {
namespace base {
    class clusters;
}

class medoid_coordinator;
class medoid : public thread {
    private:
         // Pointer to global cluster data
        std::shared_ptr<kbase::clusters> g_clusters;
        const unsigned nprocrows; // How many rows to process

        // Medoid specific
        std::vector<double> local_medoid_energy;
        std::vector<unsigned> candidate_medoids;
        std::vector<double> candidate_medoid_energy;
        const double sample_rate;
        std::default_random_engine generator;
        std::uniform_real_distribution<double> ur_distribution;
        medoid_coordinator* coord;

        // End Medoid specific

        medoid(const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol,
                std::shared_ptr<kbase::clusters> g_clusters,
                unsigned* cluster_assignments,
                const std::string fn, const double sample_rate);
    public:
        static thread::ptr create(
                const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol,
                std::shared_ptr<kbase::clusters> g_clusters,
                unsigned* cluster_assignments, const std::string fn,
                const double sample_rate) {
            return thread::ptr(
                    new medoid(node_id, thd_id, start_rid,
                        nprocrows, ncol, g_clusters,
                        cluster_assignments, fn, sample_rate));
        }

        void start(const thread_state_t state) override;
        // Allocate and move data using this thread
        void EM_step();
        void medoid_step();
        void run() override;
        void set_coordinator(medoid_coordinator* coord) {
            this->coord = coord;
        }

        // Medoid specific
        std::vector<double>& get_local_medoid_energy() {
            return local_medoid_energy;
        }

        std::vector<unsigned>& get_candidate_medoids() {
            return candidate_medoids;
        }

        std::vector<double>& get_candidate_energy() {
            return candidate_medoid_energy;
        }
        // End Medoid specific
};
}
#endif

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

#ifndef __KNOR_MEDOID_THREAD_HPP__
#define __KNOR_MEDOID_THREAD_HPP__

#include <vector>
#include "thread.hpp"

namespace knor { namespace base {
    class clusters;
} }

namespace kbase = knor::base;
namespace kprune = knor::prune;

namespace knor {
    namespace prune {
        class dist_matrix;
    }

class medoid_thread : public thread {
    private:
         // Pointer to global cluster data
        std::shared_ptr<kbase::clusters> g_clusters;
        unsigned nprocrows; // How many rows to process

        // Medoid specific
        std::shared_ptr<kprune::dist_matrix> pw_dm;
        std::vector<double> local_medoid_energy;
        double* global_medoid_energy;
        std::vector<unsigned> candidate_medoids;
        std::vector<double> candidate_medoid_energy;
        // End Medoid specific

        medoid_thread(const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol,
                std::shared_ptr<kbase::clusters> g_clusters,
                unsigned* cluster_assignments,
                const std::string fn,
                std::shared_ptr<kprune::dist_matrix> pw_dm,
                double* global_medoid_energy);
    public:
        static thread::ptr create(
                const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol,
                std::shared_ptr<kbase::clusters> g_clusters,
                unsigned* cluster_assignments, const std::string fn,
                std::shared_ptr<kprune::dist_matrix> pw_dm,
                double* global_medoid_energy) {
            return thread::ptr(
                    new medoid_thread(node_id, thd_id, start_rid,
                        nprocrows, ncol, g_clusters,
                        cluster_assignments, fn, pw_dm, global_medoid_energy));
        }

        void start(const thread_state_t state);
        // Allocate and move data using this thread
        void EM_step();
        void medoid_step();
        const unsigned get_global_data_id(const unsigned row_id) const;
        void run() override;;
        const void print_local_data() override;

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

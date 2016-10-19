/*
 * Copyright 2016 neurodata (http://neurodata.io/)
 * Written by Disa Mhembere (disa@jhu.edu)
 *
 * This file is part of k-par-means
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

#include "dist_task_thread.hpp"
#include "task_queue.hpp"
//#include "dist_task_coordinator.hpp"
#include "common.hpp"

namespace kpmeans { namespace prune {

    dist_task_thread::dist_task_thread(const int node_id, const unsigned thd_id,
            const unsigned start_rid, const unsigned nlocal_rows,
            const unsigned ncol, std::shared_ptr<kpmbase::prune_clusters> g_clusters,
            unsigned* cluster_assignments,
            const std::string fn) : kmeans_task_thread(node_id, thd_id,
                start_rid, nlocal_rows, ncol, g_clusters,
                cluster_assignments, fn) {
            }

    bool dist_task_thread::try_steal_task() {
        std::cout << "I'm in the dist_task_thread\n";
        return false; // TODO: Stub -- currently says I failed to steal a task
    }
    // TODO
} } // End namespace kpmeans::prune

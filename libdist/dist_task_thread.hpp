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

#ifndef __KPM_DIST_TASK_THREAD_HPP__
#define __KPM_DIST_TASK_THREAD_HPP__

#include <atomic>

#include "kmeans_task_thread.hpp"
#include "thread_state.hpp"

namespace kpmeans {
    namespace base {
    class prune_clusters;
    }
    namespace prune {
    }
}

namespace kpmbase = kpmeans::base;

namespace kpmeans { namespace prune {

class dist_task_thread : public kmeans_task_thread {
private:
    dist_task_thread(const int node_id, const unsigned thd_id,
            const unsigned start_rid, const unsigned nlocal_rows,
            const unsigned ncol,
            std::shared_ptr<kpmbase::prune_clusters> g_clusters,
            unsigned* cluster_assignments,
            const std::string fn);

public:
    typedef std::shared_ptr<dist_task_thread> ptr;

    static ptr create(const int node_id, const unsigned thd_id,
            const unsigned start_rid, const unsigned nlocal_rows,
            const unsigned ncol,
            std::shared_ptr<kpmbase::prune_clusters> g_clusters,
            unsigned* cluster_assignments, const std::string fn) {
        return ptr(new dist_task_thread(node_id, thd_id, start_rid,
                    nlocal_rows, ncol, g_clusters,
                    cluster_assignments, fn));
    }

    virtual bool try_steal_task();
};
} } // End namespace kpmeans, prune
#endif

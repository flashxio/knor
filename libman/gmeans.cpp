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

#include <iostream>
#include <cassert>

#include "gmeans.hpp"
#include "types.hpp"
#include "util.hpp"
#include "io.hpp"
#include "clusters.hpp"
#include "thd_safe_bool_vector.hpp"

namespace knor {

void gmeans::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback<gmeans>, this);
    if (rc)
        throw base::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

void gmeans::H_split_step() {
    for (unsigned row = 0; row < nprocrows; row++) {
        // What cluster is this row in?
        unsigned true_row_id = get_global_data_id(row);

        // Not active
        if (!(cltr_active_vec->get(cluster_assignments[true_row_id])))
            continue; // Skip it

        const size_t offset = row*ncol;
        auto const& v = std::static_pointer_cast<base::h_clusters>(
                            g_hcltrs[part_id[true_row_id]])->metadata;

        double dotprod = 0;
        for (size_t col = 0; col < ncol; col++)
            dotprod += local_data[offset+col] * v[col];

        nearest_cdist[true_row_id] = dotprod/v[ncol];
    }
}

} // End namespace knor

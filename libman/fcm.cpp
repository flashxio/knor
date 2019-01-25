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

#include "fcm.hpp"
#include "types.hpp"
#include "dense_matrix.hpp"
#include "io.hpp"
#include "util.hpp"

namespace knor {
fcm::fcm(const int node_id, const unsigned thd_id,
            const unsigned start_rid, const unsigned nprocrows,
            const unsigned ncol, const unsigned nclust,
            const unsigned fuzzindex,
            base::dense_matrix<double>* um,
            base::dense_matrix<double>* centers,
            const std::string fn, base::dist_t dist_metric) :
        thread(node_id, thd_id, ncol, NULL, start_rid, fn, dist_metric),
        nprocrows(nprocrows), centers(centers), um(um),
        nclust(nclust), fuzzindex(fuzzindex){

            this->innerprod = base::dense_matrix<double>::create(nclust, ncol);
            set_data_size(sizeof(double)*nprocrows*ncol);
    }

void fcm::Estep() {
    for (unsigned row = 0; row < nprocrows; row++) {
        unsigned true_rid = get_global_data_id(row);
        for (unsigned cid = 0; cid < nclust; cid++) {
            double dist = base::dist_comp_raw<double>(&local_data[row*ncol],
                    &(centers->as_pointer()[cid*ncol]), ncol, dist_metric);
            if (dist > 0) {
                //TODO: Fix bad access pattern. um -> col major
                um->set(cid, true_rid,
                        std::pow((1.0 / dist), (1.0 / (fuzzindex-1))));
            } else {
                um->set(cid, true_rid, 2.2E-16);
            }
        }
    }
}

// NOTE: Sequential access on all 3 matrices, but lh matrix (um) has strided
//  access.
void fcm::Mstep() {
#if 1
    innerprod->zero(); // Reset this

    for (unsigned lcid = start_rid; lcid < start_rid+nprocrows; lcid++) {
        unsigned rrid = lcid - start_rid;
        for (unsigned lrid = 0; lrid < nclust; lrid++) {
            for (unsigned rcid = 0; rcid < ncol; rcid++) {
                innerprod->peq(lrid, rcid,
                        um->get(lrid, lcid) * local_data[rrid*ncol+rcid]);
            }
        }
    }
#else
    // For testing matrix mult
    base::dense_matrix<double>* _data =
        base::dense_matrix<double>::create(nprocrows, ncol);
    _data->set(local_data);
    auto ip = ((*um) * (*_data));
    innerprod->copy_from(ip);
    innerprod->print();
#endif
}

void fcm::run() {
    switch(state) {
        case TEST:
            test();
            break;
        case ALLOC_DATA:
            numa_alloc_mem();
            break;
        case E:
            Estep();
            break;
        case M:
            Mstep();
            break;
        case EXIT:
            throw kbase::thread_exception(
                    "Thread state is EXIT but running!\n");
        default:
            throw kbase::thread_exception("Unknown thread state\n");
    }
    sleep();
}

void fcm::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback<fcm>, this);
    if (rc)
        throw kbase::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

fcm::~fcm() {
    delete (innerprod);
}
} // End namespace knor

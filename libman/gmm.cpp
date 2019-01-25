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

#include <math.h>
#include <iostream>
#include <cassert>

#include "gmm.hpp"
#include "types.hpp"
#include "util.hpp"
#include "io.hpp"
#include "linalg.hpp"

namespace knor {
gmm::gmm(const int node_id, const unsigned thd_id,
        const unsigned start_rid,
        const unsigned nprocrows, const unsigned ncol,
        const std::string fn, kbase::dist_t dist_metric) :
            thread(node_id, thd_id, ncol,
            NULL, start_rid, fn, dist_metric), nprocrows(nprocrows),
            L(0){

            local_clusters = nullptr;
            set_data_size(sizeof(double)*nprocrows*ncol);
#if VERBOSE
#ifndef
            std::cout << "Initializing gmm. Metadata: thd_id: "
                << this->thd_id << ", start_rid: " << this->start_rid <<
                ", node_id: " << this->node_id << ", nprocrows: " <<
                this->nprocrows << ", ncol: " << this->ncol << std::endl;
#endif
#endif
        }

void gmm::set_alg_metadata(unsigned k, base::dense_matrix<double>* mu_k,
        base::dense_matrix<double>** sigma_k, base::dense_matrix<double>* P_nk,
        double* Pk, base::dense_matrix<double>** isk, double* dets,
        double* Px) {
    this->k = k;
    this->mu_k = mu_k;
    this->sigma_k = sigma_k;
    this->P_nk = P_nk;
    inv_sigma_k = isk;
    this->dets = dets;
    this->Pk = Pk;
    this->Px = Px;
}

void gmm::Estep() {
    // Compute Pnk
    L = 0; // Reset
    for (unsigned row = 0; row < nprocrows; row++) {
        unsigned true_row_id = get_global_data_id(row);

        for (unsigned cid = 0; cid < k; cid++) {
            std::vector<double> diff(ncol);
            base::linalg::vdiff(&local_data[row*ncol],
                    &(mu_k->as_pointer()[row*ncol]), ncol, diff);

            std::vector<double> resdot(inv_sigma_k[cid]->get_ncol());
            base::linalg::dot(&diff[0],
                    inv_sigma_k[cid]->as_pointer(),
                    inv_sigma_k[cid]->get_nrow(),
                    inv_sigma_k[cid]->get_ncol(), resdot);
            double lhs = -.5*(base::linalg::dot(resdot, diff));
            double rhs = .5*M*std::log2(2*M_PI) - (.5*std::log2(dets[cid]));
            double gaussian_density = lhs - rhs; // for one of the k guassians

            auto tmp = gaussian_density * Pk[cid];
            Px[true_row_id] += tmp;
            P_nk->set(row, cid, tmp);
        }

        // Finish P_nk
        for (unsigned cid = 0; cid < k; cid++) {
            P_nk->set(row, cid, P_nk->get(row, cid)/Px[true_row_id]);
        }

        // L is Per thread
        if (L == 0) L = Px[true_row_id];
        else L *= Px[true_row_id];
    }
}

void gmm::Mstep() {

}

void gmm::run() {
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

void gmm::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback<gmm>, this);
    if (rc)
        throw kbase::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}
} // End namespace knor

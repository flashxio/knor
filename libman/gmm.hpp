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

#ifndef __KNOR_GMM_HPP__
#define __KNOR_GMM_HPP__

#include "thread.hpp"

namespace knor { namespace base {
    template <typename T> class dense_matrix;
} }

namespace kbase = knor::base;

namespace knor {
class gmm : public thread {
    protected:
         // Pointer to global cluster data
        unsigned nprocrows; // How many rows to process
        unsigned k;
        base::dense_matrix<double>* mu_k;
        base::dense_matrix<double>* local_mu_k;
        base::dense_matrix<double>** sigma_k;
        base::dense_matrix<double>* P_nk;
        double* Pk;
        double* dets; // The determinant of each covariance matrix
        base::dense_matrix<double>** inv_sigma_k; // Inverse of sigma
        double* Px;
        double L;

        gmm(const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol,
                const std::string fn, kbase::dist_t dist_metric);
    public:
        static thread::ptr create(
                const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol,
                const std::string fn,

                kbase::dist_t dist_metric) {
            return thread::ptr(
                    new gmm(node_id, thd_id, start_rid,
                        nprocrows, ncol, fn, dist_metric));
        }

        void set_alg_metadata(unsigned k, base::dense_matrix<double>* mu_k,
                base::dense_matrix<double>** sigma_k,
                base::dense_matrix<double>* P_nk, double* Pk,
                base::dense_matrix<double>** isk, double* dets, double* Px);

        double get_L() { return L; }
        void start(const thread_state_t state) override;
        // Allocate and move data using this thread
        void Estep();
        void Mstep();
        virtual void run() override;
};
}
#endif

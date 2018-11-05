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
#ifndef __KNOR_GMM_COORDINATOR_HPP__
#define __KNOR_GMM_COORDINATOR_HPP__

#include "coordinator.hpp"
#include "util.hpp"

namespace knor {

namespace base {
    template <typename T> class dense_matrix;
}

class gmm_coordinator : public coordinator {
    protected:
        // Metadata
        // max index stored within each threads partition

        base::dense_matrix<double>* mu_k; // estimated guassians (k means)
        std::vector<base::dense_matrix<double>*> sigma_k; // k Covar matrices
        std::vector<base::dense_matrix<double>*> inv_sigma_k; // k Covar matrices
        std::vector<double> dets; // Determinants
        base::dense_matrix<double>* P_nk; // responsibility matrix (nxk)
        std::vector<double> Pk; // Frac of points in component k
        double cov_regularizer;
        std::vector<double> Px;
        std::vector<double> Pnk_sum; // 1 x k
        unsigned k;
        double L;

        gmm_coordinator(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k,
                const unsigned max_iters, double* mu_k,
                const unsigned nnodes, const unsigned nthreads,
                const base::init_t it,
                const double tolerance, const base::dist_t dt,
                const double cov_regularizer);

    public:
        static coordinator::ptr create(const std::string fn,
                const size_t nrow, const size_t ncol, const unsigned k,
                const unsigned max_iters, double* mu_k,
                const unsigned nnodes, const unsigned nthreads,
                const std::string init="forgy", const double tolerance=-1,
                const std::string dist_type="eucl",
                const double cov_regularizer=1E-6) {

            base::init_t _init_t = base::get_init_type(init);
            base::dist_t _dist_t = base::get_dist_type(dist_type);
#if KM_TEST
#ifndef BIND
            printf("gmm coordinator => NUMA nodes: %u, nthreads: %u, "
                    "nrow: %lu, ncol: %lu, init: '%s', dist_t: '%s', fn: '%s'"
                    "\n\n", nnodes, nthreads, nrow, ncol, init.c_str(),
                    dist_type.c_str(), fn.c_str());
#endif
#endif
            return coordinator::ptr(
                    new gmm_coordinator(fn, nrow, ncol, k, max_iters,
                    mu_k, nnodes, nthreads, _init_t, tolerance, _dist_t,
                    cov_regularizer));
        }

        // Pass file handle to threads to read & numa alloc
        virtual base::cluster_t run(double* allocd_data=NULL,
            const bool numa_opt=false) override {
            throw base::not_implemented_exception();
        };

        void compute_cov_mat();
        base::gmm_t soft_run(double* allocd_data=NULL);
        void compute_shared_linalg();
        void random_prob_fill(base::dense_matrix<double>* dm,
                const double mix=0, const double max=1);
        void random_prob_fill(std::vector<double>& v,
                const double min=0, const double max=1);
        void update_clusters();
        void random_partition_init() override;
        void forgy_init() override;
        void kmeanspp_init() override;
        virtual void preprocess_data() {
            throw base::not_implemented_exception();
        }
        virtual void build_thread_state() override;
        ~gmm_coordinator();
};
} // End namespace knor
#endif

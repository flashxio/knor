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

#ifndef __KNOR_FCM_HPP__
#define __KNOR_FCM_HPP__

#include "thread.hpp"

namespace knor { namespace base {
    template <typename T> class dense_matrix;
} }

namespace knor {
class fcm : public thread {
    protected:
        unsigned nprocrows; // How many rows to process
        base::dense_matrix<double>* centers;
        base::dense_matrix<double>* um; // Contribution matrix
        base::dense_matrix<double>* innerprod;
        unsigned nclust;
        unsigned fuzzindex;

        fcm(const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol,
                const unsigned nclust,
                const unsigned fuzzindex,
                base::dense_matrix<double>* um,
                base::dense_matrix<double>* centers,
                // Partition of result matrix of um.dot(data)
                const std::string fn, base::dist_t dist_metric);
    public:
        static thread::ptr create(
                const int node_id, const unsigned thd_id,
                const unsigned start_rid, const unsigned nprocrows,
                const unsigned ncol, const unsigned nclust,
                const unsigned fuzzindex, base::dense_matrix<double>* um,
                base::dense_matrix<double>* centers,
                const std::string fn, base::dist_t dist_metric) {
            return thread::ptr(
                    new fcm(node_id, thd_id, start_rid,
                        nprocrows, ncol, nclust, fuzzindex, um,
                        centers, fn, dist_metric));
        }

        void start(const thread_state_t state) override;
        // Allocate and move data using this thread
        void Estep();
        void Mstep();

        void run() override;
        base::dense_matrix<double>* get_innerprod() {
            return innerprod;
        }
        ~fcm();
};
}
#endif

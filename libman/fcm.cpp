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

namespace knor {
fcm::fcm(const int node_id, const unsigned thd_id,
            const unsigned start_rid, const unsigned nprocrows,
           const unsigned ncol, const unsigned nclust,
            const unsigned fuzz_idx,
            const std::string fn, base::dist_t dist_metric) :
        thread(node_id, thd_id, ncol, NULL, start_rid, fn, dist_metric) {

            this->fuzz_idx = fuzz_idx;
            this->nclust = nclust;
            this->nprocrows = nprocrows;

            set_data_size(sizeof(double)*nprocrows*ncol);
#if VERBOSE
#ifndef
            printf("Initializing fcm. Metadata: thd_id: %u , "
                    "start_rid: %u, node_id: %d, nprocrows: %u, ncol: %u\n",
                    this->thd_id, this->start_rid, this->node_id,
                    this->nprocrows, this->ncol);
#endif
#endif
    }

void fcm::Estep() {
    // TODO
}

void fcm::Mstep() {
    // TODO
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

void* callback(void* arg) {
    fcm* t = static_cast<fcm*>(arg);
#ifdef USE_NUMA
    t->bind2node_id();
#endif

    while (true) { // So we can receive task after task
        if (t->get_state() == WAIT)
            t->wait();

        if (t->get_state() == EXIT) {// No more work to do
            //printf("Thread %d exiting ...\n", t->thd_id);
            break;
        }

        //printf("Thread %d awake and doing a run()\n", t->thd_id);
        t->run(); // else
    }

    // We've stopped running so exit
    pthread_exit(NULL);

#ifdef _WIN32
    return NULL;
#endif
}

void fcm::start(const thread_state_t state=WAIT) {
    this->state = state;
    int rc = pthread_create(&hw_thd, NULL, callback, this);
    if (rc)
        throw kbase::thread_exception(
                "Thread creation (pthread_create) failed!", rc);
}

const void fcm::print_local_data() {
    kbase::print_mat(local_data, nprocrows, ncol);
}
} // End namespace knor

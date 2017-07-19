/**
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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <pthread.h>
#include <atomic>
#include <iostream>

#include "kmeans_thread.hpp"
#include "thread_state.hpp"
#include "clusters.hpp"
#include "io.hpp"
#include "util.hpp"

#ifdef USE_NUMA
#include "numa.h"
#endif

namespace kpmbase = kpmeans::base;

static std::atomic<unsigned> pending_threads;
//static unsigned pending_threads;
static pthread_mutex_t mutex;
static pthread_cond_t cond;
static pthread_mutexattr_t mutex_attr;

static void wait4complete() {
    //printf("\nParent entering wait4complete ..\n");
    pthread_mutex_lock(&mutex);
    while (pending_threads != 0) {
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_mutex_unlock(&mutex);
    //printf("Exiting wait4complete!!\n\n");
}

static void wake4run(std::vector<kpmeans::kmeans_thread::ptr>& threads,
        const unsigned nthreads, const kpmeans::thread_state_t state) {
    pending_threads = nthreads;
    for (unsigned thd_id = 0; thd_id < threads.size(); thd_id++) {
        threads[thd_id]->wake(state);
    }
}

static void test_thread_creation(const unsigned NTHREADS,
        const unsigned nnodes) {
    std::vector<kpmeans::kmeans_thread::ptr> threads;

    // Always: Build state alone
    for (unsigned i = 0; i < NTHREADS; i++) {
        kpmbase::clusters::ptr cl = kpmbase::clusters::create(2,2);
        threads.push_back(kpmeans::kmeans_thread::create
                (i%nnodes, i, 69, 200, 1, cl, NULL, "/dev/null"));
        threads[i]->set_parent_cond(&cond);
        threads[i]->set_parent_pending_threads(&pending_threads);
        // Thread puts itself to sleep
        threads[i]->start(kpmeans::thread_state_t::WAIT);
    }

    for (unsigned i = 0; i < 2048; i++) {
        wake4run(threads, NTHREADS, kpmeans::thread_state_t::TEST);
        wait4complete();
    }

    wake4run(threads, NTHREADS, kpmeans::thread_state_t::EXIT);
    std::cout << "SUCCESS: for creation & join\n";
}

void test_numa_populate_data(const unsigned NTHREADS, const unsigned nnodes,
    const size_t nrow=50, const size_t ncol=5,
    const std::string fn="../../test-data/matrix_r50_c5_rrw.bin") {

    printf("\nRunning test_numa_populate_data with "
            "%u threads ...\n", NTHREADS);
    const unsigned nprocrows = nrow/NTHREADS;

    std::vector<kpmeans::kmeans_thread::ptr> threads;

    // Always: Build state alone
    for (unsigned i = 0; i < NTHREADS; i++) {
        kpmbase::clusters::ptr cl = kpmbase::clusters::create(2,2);
        threads.push_back(kpmeans::kmeans_thread::create
                (i%nnodes, i, i*nprocrows, nprocrows, ncol,
                 cl, NULL, fn));
        threads[i]->set_parent_cond(&cond);
        threads[i]->set_parent_pending_threads(&pending_threads);
        // Thread puts itself to sleep
        threads[i]->start(kpmeans::thread_state_t::WAIT);
    }

    kpmeans::base::bin_io<double> br(fn, nrow, ncol);
    double* data = new double [nrow*ncol];
    printf("Bin read data\n");
    br.read(data);

    wake4run(threads, NTHREADS, kpmeans::thread_state_t::ALLOC_DATA);
    wait4complete();

    std::vector<kpmeans::kmeans_thread::ptr>::iterator it = threads.begin();
    // Print it back
    for (it = threads.begin(); it != threads.end(); ++it) {
        double *dp = &data[(*it)->get_thd_id()*ncol*nprocrows];
        assert(kpmbase::eq_all(dp, (*it)->get_local_data(),
                    nprocrows*ncol));
        printf("Thread %u PASSED numa_mem_alloc()\n", (*it)->get_thd_id());
    }

    wake4run(threads, NTHREADS, kpmeans::thread_state_t::EXIT);
    delete [] data;
    printf("SUCCESS test_numa_populate_data ..\n");
}

int main(int argc, char* argv[]) {
    pending_threads = 0; // NOTE: This must be initialized
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_init(&mutex, &mutex_attr);
    pthread_cond_init(&cond, NULL);
    unsigned nnodes = kpmbase::get_num_nodes();

    if (argc < 2) {
        fprintf(stderr, "usage: ./test_kmeans_thread nthreads [nnodes]\n");
        exit(EXIT_FAILURE);
    }
    const unsigned nthreads = atoi(argv[1]);

    if (argc > 2) {
        if (atol(argv[2]) <= nnodes) {
            std::cout << "[NOTE]: Setting NUMA nodes to: " << argv[2] <<
                ". The max is: " << nnodes << std::endl;
            nnodes = atoi(argv[2]);
        } else {
            std::cout << "[WARNING]: Rejected excess request of NUMA nodes of: " <<
                argv[2] << "\n\n";
        }
    }

    std::cout << "Test begins with: " << nthreads << " threads, " << nnodes <<
        " numa nodes.\n";

    test_thread_creation(nthreads, nnodes);
    test_numa_populate_data(nthreads, nnodes);

    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&mutex);
    pthread_mutexattr_destroy(&mutex_attr);

    return (EXIT_SUCCESS);
}

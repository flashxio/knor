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

#include "task_queue.hpp"
#include "io.hpp"
#include "util.hpp"
#ifdef USE_NUMA
#include "numa.h"
#endif

void test_queue_get(const unsigned NTHREADS, const unsigned nnodes,
    const size_t nrow=50, const size_t ncol=5,
    const std::string fn="../../test-data/matrix_r50_c5_rrw.bin") {

    printf("\nRunning test_queue_get with"
            " constexpr NTHREADS = %u...\n", NTHREADS);

    kpmbase::bin_io<double> br(fn, nrow, ncol);
    double* data = new double [nrow*ncol];
    printf("Bin read data\n");
    br.read(data);

    kpmeans::task_queue q(data, 0, nrow, ncol);
    printf("Task queue ==> nrow: %u, ncol: %u\n",
            q.get_nrow(), q.get_ncol());

    printf("run:");
    for (unsigned i = 0; i < 4; i++) {
        printf(" %u", i);
        // Test reset
        q.reset();
        while(q.has_task()) {
            kpmeans::task* t = q.get_task();
            assert(kpmbase::eq_all<double>(
                        t->get_data_ptr(), &(data[t->get_start_rid()*ncol]),
                        t->get_nrow()*ncol));
            delete t;
        }
    }

    printf("\n\nTask queue test SUCCESSful! ...\n");
    delete [] data;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: ./test_task_queue nthreads [nnodes]\n");
        exit(EXIT_FAILURE);
    }

    unsigned nnodes = kpmbase::get_num_nodes();
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

    test_queue_get(atol(argv[1]), nnodes);
    return (EXIT_SUCCESS);
}

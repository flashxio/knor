/**
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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <iostream>

#include <cassert>

#include "dist_matrix.hpp"
#include "io.hpp"

namespace clustercore = knor::core;
namespace kprune = knor::prune;

constexpr unsigned NROW = 16;
constexpr unsigned NCOL = 4;

void test_dist_matrix() {
    std::vector<double> data(NROW*NCOL); // It's dense
    clustercore::bin_rm_reader<double> bm0("data_dm.bin");
    bm0.read(data);

    auto dm = kprune::dist_matrix::create(NROW);
    dm->compute_pairwise_dist(&data[0], NCOL, clustercore::dist_t::TAXI);

    std::vector<double> dense_dm(NROW*NROW); // It's dense
    clustercore::bin_rm_reader<double> bm1("pw_dm.bin");
    bm1.read(dense_dm);

    for (unsigned row = 0; row < NROW; row++) {
        for (unsigned col = row + 1; col < NROW; col++) {
            assert(dense_dm[(row*NROW)+col] == dm->pw_get(row, col));
        }
    }
}

int main() {
    test_dist_matrix();
    printf("Successful 'test_dist_matrix' test ...\n");
    return EXIT_SUCCESS;
}
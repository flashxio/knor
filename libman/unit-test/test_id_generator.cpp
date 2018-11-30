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

#include <omp.h>

#include <iostream>
#include <cassert>

#include "hclust_id_generator.hpp"

int main(int argc, char* argv []) {

    auto geny = knor::hclust_id_generator::create(); // Smart pointer type
    geny->print(); // Expecting 0 -> (1,2)

    // TODO: Dispatch each job to a thread
    auto kv = geny->get_split_ids(0);
    assert(kv.first == 1);
    assert(kv.second == 2);
    geny->print(); // Expecting 0 -> (1,2)

    // Test for 1
    kv = geny->get_split_ids(1);
    assert(kv.first == 3);
    assert(kv.second == 4);
    geny->print(); // Expecting 0 -> (1,2)  1 -> (3,4)

    // Test for 2
    kv = geny->get_split_ids(2);
    assert(kv.first == 5);
    assert(kv.second == 6);
    geny->print(); // Expecting 0 -> (1,2) 1 -> (3,4) 2 -> (5, 6)

    // Test for 2 again
    kv = geny->get_split_ids(2);
    assert(kv.first == 5);
    assert(kv.second == 6);

    std::cout << "Sucessful ID generator test!\n";
    return EXIT_SUCCESS;
}

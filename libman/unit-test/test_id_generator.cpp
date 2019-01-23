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
#include <vector>

#include "hclust_id_generator.hpp"

using namespace knor;

void check_insert(unsigned parent,
        std::unordered_map<unsigned, unsigned>& map, split_id& lr) {

    if (map.find(lr.first) != map.end()) {
        printf("Invalid re-entry for parent: %u, child: %u, exists with "
                "parent: %u\n", parent, lr.first, map[lr.first]);
        assert(0);
    }
    map[lr.first] = parent;
    printf("Inserting child: %u, parent: %u\n", lr.first, parent);

    if (map.find(lr.second) != map.end()) {
        printf("Invalid re-entry for parent: %u, child: %u, exists with "
                "parent: %u\n", parent, lr.second, map[lr.second]);
        assert(0);
    }
    map[lr.second] = parent;
    printf("Inserting child: %u, parent: %u\n", lr.second, parent);
}

int main(int argc, char* argv []) {

    auto geny = hclust_id_generator::create(); // Smart pointer type
    geny->print(); // Expecting 0 -> (1,2)

    // <child, parent>
    std::unordered_map<unsigned, unsigned> used;

    // TODO: Dispatch each job to a thread
    auto lr = geny->get_split_ids(0);
    check_insert(0, used, lr);

    assert(lr.first == 1);
    assert(lr.second == 2);
    geny->print(); // Expecting 0 -> (1,2)

    // Test for 1
    lr = geny->get_split_ids(1);
    check_insert(1, used, lr);
    assert(lr.first == 3);
    assert(lr.second == 4);
    geny->print(); // Expecting 0 -> (1,2)  1 -> (3,4)

    // Test for 2
    lr = geny->get_split_ids(2);
    check_insert(2, used, lr);
    assert(lr.first == 5);
    assert(lr.second == 6);
    geny->print(); // Expecting 0 -> (1,2) 1 -> (3,4) 2 -> (5, 6)

    // Test for 2 again
    lr = geny->get_split_ids(2);
    assert(lr.first == 5);
    assert(lr.second == 6);

    std::vector<unsigned> new_ids {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18};
    for (auto const& entry : new_ids) {
        lr = geny->get_split_ids(entry);
        check_insert(entry, used, lr);
    }

    std::cout << "Sucessful ID generator test!\n";
    return EXIT_SUCCESS;
}
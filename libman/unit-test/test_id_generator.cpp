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

#include <cassert>
#include "hclust_id_generator.hpp"

using namespace knor;

int main(int argc, char* argv []) {
    auto geny = hclust_id_generator::create(); // Smart pointer type

    constexpr unsigned TESTSIZE = 5;
    const unsigned MAXID = (TESTSIZE*2) + 1;

    std::vector<split_id> gend_ids;
    for (size_t i = 0; i < TESTSIZE; i++)
        gend_ids.push_back(geny->get_split_ids());

    for (auto p : gend_ids)
        assert(p.first < MAXID && p.second < MAXID);

    printf("before reclaim:\n");
    geny->print();

    // Reclaim all
    for (auto pair : gend_ids) {
        geny->reclaim_id(pair.first);
        geny->reclaim_id(pair.second);
    }

    printf("after reclaim:\n");
    geny->print();

    // Clear all
    gend_ids.clear();
    // Repopulate
    for (size_t i = 0; i < TESTSIZE; i++)
        gend_ids.push_back(geny->get_split_ids());

    printf("after repop:\n");
    geny->print();

    for (auto p : gend_ids)
        assert(p.first < MAXID && p.second < MAXID);

    std::cout << "Sucessful ID generator test!\n";
    return EXIT_SUCCESS;
}

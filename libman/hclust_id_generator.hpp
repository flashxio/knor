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
#ifndef __KNOR_HCLUST_ID_GENERATOR_HPP__
#define __KNOR_HCLUST_ID_GENERATOR_HPP__

#include <unordered_map>
#include <memory>
#include <stdexcept>
#include "io.hpp"

namespace knor {

struct split_id {
    unsigned first, second;
    split_id() { }

    split_id(unsigned _first, unsigned _second):
        first(_first), second(_second) {
    }
};

// Expected to be used as a Singleton class
class hclust_id_generator {
private:
    unsigned max_id;
    std::vector<unsigned> recycler;

    hclust_id_generator() : max_id(0) {
    }

public:
    typedef std::shared_ptr<hclust_id_generator> ptr;

    // Reclaim single IDs
    void reclaim_id(const unsigned id) {
        recycler.push_back(id);
    }

    const void print() const {
#ifndef BIND
        printf("Printing hclust_id_generator: Max ID: %u\n", max_id);
        printf("recycler: \n"); base::print(recycler);
#endif
    }

    static ptr create() {
        return ptr(new hclust_id_generator());
    };

    // Given a cluster id to which a node is assigned, get the split IDs
    split_id get_split_ids() {
        // Increments max_id
        unsigned l, r;
        if (recycler.size()) {
            l = recycler.back();
            recycler.pop_back();
        } else {
            l = ++max_id;
        }

        if (recycler.size()) {
            r = recycler.back();
            recycler.pop_back();
        } else {
            r = ++max_id;
        }
        return split_id(l, r);
    }
};
}
#endif

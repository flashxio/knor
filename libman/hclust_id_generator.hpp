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

// This will be slow, but will allow for testing
#if 0
#include <shared_mutex>
#else
#include <mutex>
#endif

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
    std::unordered_map<unsigned, split_id> id_split_map;
#if 0
    mutable std::shared_mutex _mutex;
#else
    mutable std::mutex _mutex;
#endif

    hclust_id_generator() {
#if 0
        _mutex.lock();
#else
        _mutex.lock();
#endif
        max_id = 2;
        id_split_map[0] = split_id(1, 2);
#if 0
        _mutex.unlock();
#else
        _mutex.unlock();
#endif
    }

    // Method is only called with exclusive lock taken
    void generate_next(unsigned id) {
        // Increments max_id
        auto l = ++max_id;
        auto r = ++max_id;
        id_split_map[id] = split_id(l, r);
    }

public:
    typedef std::shared_ptr<hclust_id_generator> ptr;

    const void print() const {
        printf("Printing hclust_id_generator:\n");
        for (auto kv : id_split_map) {
            printf("%u -> (%u, %u)\n", kv.first,
                    kv.second.first, kv.second.second);
        }
    }

    static ptr create() {
        return ptr(new hclust_id_generator());
    };

    // Given a cluster id to which a node is assigned, get the split IDs
    split_id& get_split_ids(unsigned id) {
        // Readers lock
#if 0
        _mutex.lock_shared();
#else
        _mutex.lock();
#endif
        auto entry = id_split_map.find(id);
        if (entry != id_split_map.end()) {
#if 0
            _mutex.unlock_shared();
#else
            _mutex.unlock();
#endif
            return entry->second; // i.e. the split_id struct
        }
#if 0
        _mutex.unlock_shared();
#else
        _mutex.unlock();
#endif

        // else
        // Acquire an exclusive lock for writing
#if 0
        _mutex.lock();
#else
        _mutex.lock();
#endif
        // Check no one else has updated the map between your check
        entry = id_split_map.find(id);
        if (entry != id_split_map.end()) {
#if 0
            _mutex.unlock();
#else
            _mutex.unlock();
#endif
            return entry->second; // i.e. the split_id struct
        } else {
            generate_next(id);
#if 0
            _mutex.unlock();
#else
            _mutex.unlock();
#endif
            return id_split_map[id]; // i.e. the split_id struct
        }

        throw std::runtime_error("Undefined state in get_split_ids!\n");
    }
};
}
#endif

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

#ifndef __KNOR_BIND_NUMA_REORG__
#define __KNOR_BIND_NUMA_REORG__

#include <numa.h>
#include <string.h>

#include <vector>
#include <memory>
#include <iostream>
#include <cassert>

#include "thread.hpp"

namespace knor { namespace binding {

/**
  * Parition data into number of NUMA nodes chunks
  */
template <typename T>
class memory_distributor {

private:
    // New pointers for data <data_ptr, numa_node>
    std::vector<T*> numa_allocd_ptrs;
    std::pair<size_t, size_t> part_size; // # ROWS in <0..n-1, n> partitions
    T* mallocd_data;
    size_t npart;
    size_t ncol;
    size_t gnrow;

    /**
      * npart - the number of NUMA nodes generally
      */
    memory_distributor(T* mallocd_data, const size_t npart,
            const size_t nrow, const size_t ncol) {
        this->npart = npart;
        this->ncol = ncol;
        this->gnrow = nrow;
        this->mallocd_data = mallocd_data;
        assert(mallocd_data);

        part_size = std::pair<size_t, unsigned>(nrow/npart,
                (nrow/npart + nrow%npart));

        assert(nrow ==
                ((part_size.first*(npart-1)) + part_size.second));
    }

public:
    typedef typename std::shared_ptr<memory_distributor> ptr;

    static ptr create(T* mallocd_data, const unsigned npart,
            const size_t nrow, const size_t ncol) {
        return ptr(new memory_distributor<T>(mallocd_data, npart, nrow, ncol));
    }

    void numa_reorg (std::vector<thread::ptr>& threads) {
        for (size_t tid = 0; tid < threads.size(); tid++) {
            size_t nbytes = threads[tid]->get_data_size();
            size_t start_rid = threads[tid]->get_start_rid();
            unsigned node_id = threads[tid]->get_node_id();
            size_t offset = start_rid*ncol;

#if VERBOSE
#ifndef BIND
            std::cout << "Thread: " <<  tid << ", nbytes: " << nbytes << ", start_rid: "
                << start_rid << ", node_id: " << node_id << ", global offset: " << offset << "\n\n";
#endif
#endif

            T* numa_allocd_data = static_cast<T*>(numa_alloc_onnode(nbytes, node_id));
            assert(numa_allocd_data);
            std::copy(&mallocd_data[offset], &mallocd_data[offset+(nbytes/sizeof(T))],
                numa_allocd_data); // TODO: Limit mem
            threads[tid]->set_local_data_ptr(numa_allocd_data, false);
        }
    }

    const size_t local_rid (const size_t global_rid,
            const unsigned node_id) const {
        size_t ret = global_rid - (node_id * (gnrow/npart));
#if VERBOSE
#ifndef BIND
        std::cout << "Given global_rid: " << global_rid << ", node_id: " << node_id
            << " ==> local_rid: " << ret << "\n";
#endif
#endif
        return ret;
    }

    std::vector<T*>& get_ptrs() {
        return numa_allocd_ptrs;
    }

    ~memory_distributor() {
        // Dealloc happens in each thread ...
    }
};
}} // End namespace knor::binding

#endif

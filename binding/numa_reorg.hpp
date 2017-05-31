/*
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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY CURRENT_KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __KPM_NUMA_REORG__
#define __KPM_NUMA_REORG__

#include <numa.h>
#include <string.h>

#include <vector>
#include <memory>
#include <iostream>

#include <boost/assert.hpp>
#include "../libman/base_kmeans_thread.hpp"

namespace kpmeans { namespace binding {

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

    /**
      * npart - the number of NUMA nodes generally
      */
    memory_distributor(T* mallocd_data, const size_t npart,
            const size_t nrow, const size_t ncol) {
        this->npart = npart;
        this->ncol = ncol;
        this->mallocd_data = mallocd_data;
        BOOST_VERIFY(mallocd_data);

        part_size = std::pair<size_t, unsigned>(nrow/npart,
                (nrow/npart + nrow%npart));

        BOOST_VERIFY(nrow ==
                ((part_size.first*(npart-1)) + part_size.second));
    }

public:
    typedef typename std::shared_ptr<memory_distributor> ptr;

    static ptr create(T* mallocd_data, const unsigned npart,
            const size_t nrow, const size_t ncol) {
        return ptr(new memory_distributor<T>(mallocd_data, npart, nrow, ncol));
    }

    void numa_reorg () {
        numa_allocd_ptrs.resize(npart);

#pragma omp parallel for
        for (size_t part_id = 0; part_id < npart; part_id++) {
            size_t nelem = (part_id == npart-1) ? part_size.second*ncol :
                part_size.first*ncol;
            size_t offset = part_id*part_size.first*ncol;

            numa_allocd_ptrs[part_id] =
                static_cast<T*>(numa_alloc_onnode(nelem*sizeof(T), part_id));
            BOOST_VERIFY(numa_allocd_ptrs[part_id]);

            std::copy(&mallocd_data[offset], &mallocd_data[offset+nelem],
                numa_allocd_ptrs[part_id]); // Limit mem?
        }
    }

    std::vector<T*>& get_ptrs() {
        return numa_allocd_ptrs;
    }

    ~memory_distributor() {
        // Dealloc happens in each thread ...
    }
};
}} // End namespace kpmeans::binding

#endif

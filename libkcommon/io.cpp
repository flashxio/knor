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

#include "io.hpp"

namespace knor { namespace base {

void store_cluster(const unsigned id, const double* data,
        const unsigned numel, const unsigned* cluster_assignments,
        const size_t nrow, const size_t ncol, const std::string dir) {
#ifndef BIND
    std::cout << "Storing cluster " << id << std::endl;
#endif

    FILE* f = nullptr;
    std::string fn = dir+"cluster_"+std::to_string(id)+
        "_r"+std::to_string(numel)+"_c"+std::to_string(ncol)+".bin";
    f = fopen(fn.c_str(), "wb");
    assert(f);
#ifndef BIND
    std::cout << "[Warning]: Writing cluster file '" <<
        fn << "'\n";
#endif
    unsigned count = 0;

    for(unsigned i = 0; i < nrow; i++) {
        if (count == numel) { break; }
        if (cluster_assignments[i] == id) {
#ifdef NDEBUG
            fwrite(&data[i*ncol],
                        (ncol*sizeof(double)), 1, f);
#else
           assert(fwrite(&data[i*ncol],
                        (ncol*sizeof(double)), 1, f));
#endif
            count++;
        }
    }
    assert(count == numel);
    fclose(f);
}
} } // End namespace knor, base
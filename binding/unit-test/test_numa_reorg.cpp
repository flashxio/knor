/*
 * Copyright 2016 neurodata (http://neurodata.io/)
 * Written by Disa Mhembere (disa@jhu.edu)
 *
 * This file is part of knor.
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

#include "numa_reorg.hpp"
#include "io.hpp"
#include "util.hpp"

namespace kpmbase = kpmeans::base;
namespace kpmbind = kpmeans::binding;

int main(int argc, char* argv[]) {

    constexpr size_t nrow = 50;
    constexpr size_t ncol = 5;
    constexpr unsigned nthread = 2;
    const std::string fn = "../../test-data/matrix_r50_c5_rrw.bin";

    std::vector<double> data(nrow*ncol);
    kpmbase::bin_rm_reader<double> br(fn);
    br.read(data);

    std::cout << "Original matrix\n";
    kpmbase::print_mat<double>(&data[0], nrow, ncol);

    unsigned nnodes = static_cast<unsigned>(numa_num_task_nodes());
    kpmbind::memory_distributor<double>::ptr md =
        kpmbind::memory_distributor<double>::create(&data[0],
                nnodes, nrow, ncol);

    md->numa_reorg();
    std::vector<double*> numa_ptrs = md->get_ptrs();

    for (size_t node_id = 0; node_id < numa_ptrs.size(); node_id++) {
        size_t prows = node_id < numa_ptrs.size() - 1 ? (nrow/nnodes) :
            ((nrow/nnodes)+(nrow%nnodes));
        std::cout << "Part: " << node_id << "\n";
        kpmbase::print_mat<double>(numa_ptrs[node_id], prows, ncol);

        BOOST_VERIFY(kpmbase::eq_all(&data[node_id*prows],
                    numa_ptrs[node_id], prows*ncol));
    }
    return EXIT_SUCCESS;
}

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

#ifndef __KPM_TEST_SHARED_HPP__
#define __KPM_TEST_SHARED_HPP__

#include <string>
#include <iostream>

#include "io.hpp"

namespace kpmbase = kpmeans::base;

namespace kpmeans { namespace test {
    const std::string TESTDATA_FN = "../test-data/matrix_r50_c5_rrw.bin";
    const std::string TEST_INIT_CLUSTERS =
        "../test-data/init_clusters_k8_c5.bin";
    const std::string TEST_CONVERGED_INIT_RES =
        "../test-data/converged_result_init_k8_c5.bin";
    constexpr size_t TEST_NROW = 50;
    constexpr size_t TEST_NCOL = 5;
    constexpr unsigned TEST_K = 8;
    constexpr double TEST_TOL = 1E-6;

    static void load_result(double* buff) {
        kpmbase::bin_io<double> br(TEST_CONVERGED_INIT_RES, TEST_K, TEST_NCOL);
        br.read(buff);
    }
} }
#endif

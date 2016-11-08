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

#include "kmeans.hpp"
#include "test_shared.hpp"
#include "util.hpp"

namespace kpmtest = kpmeans::test;

namespace kpmeans { namespace test {

kpmbase::kmeans_t test_inited(const bool prune) {
    double* p_centers = new double [TEST_K*TEST_NCOL];
    double* p_data = new double [TEST_NROW*TEST_NCOL];
    size_t* p_clust_asgn_cnt = new size_t [TEST_K];
    unsigned* p_clust_asgns = new unsigned [TEST_NROW];
    constexpr unsigned MAX_ITER = 10;
    constexpr unsigned NTHREADS = 2;

    {
    kpmbase::bin_io<double> br(TEST_INIT_CLUSTERS, TEST_K, TEST_NCOL);
    br.read(p_centers); } {
    kpmbase::bin_io<double> br(TESTDATA_FN, TEST_NROW, TEST_NCOL);
    br.read(p_data);
    }

    kpmbase::kmeans_t ret;
    if (prune) {
        ret = kpmeans::omp::compute_min_kmeans(
                p_data, p_centers, p_clust_asgns,
                p_clust_asgn_cnt, TEST_NROW, TEST_NCOL, TEST_K, MAX_ITER,
                NTHREADS, "none", 0);
    } else {
        ret = kpmeans::omp::compute_kmeans(
                p_data, p_centers, p_clust_asgns,
                p_clust_asgn_cnt, TEST_NROW, TEST_NCOL, TEST_K, 10,
                2, "none", 0);
    }

    delete [] p_centers;
    delete [] p_data;
    delete [] p_clust_asgn_cnt;
    delete [] p_clust_asgns;

    return ret;
}
} }

int main(int argc, char* argv[]) {
    double* res = new double [kpmtest::TEST_K*kpmtest::TEST_NCOL];
    kpmtest::load_result(res);

    // Auto only
    {
        kpmbase::kmeans_t ret = kpmeans::test::test_inited(false);
        BOOST_VERIFY(kpmtest::check_collection_equal(
                    ret.centroids.begin(), ret.centroids.end(),
                    res, res + kpmtest::TEST_K*kpmtest::TEST_NCOL,
                    kpmtest::TEST_TOL));
        BOOST_LOG_TRIVIAL(info) << "\n***Auto inited passed ***\n";
    }

    {
    kpmbase::kmeans_t ret = kpmeans::test::test_inited(false);
    BOOST_VERIFY(kpmtest::check_collection_equal(
                ret.centroids.begin(), ret.centroids.end(),
                res, res + kpmtest::TEST_K*kpmtest::TEST_NCOL,
                kpmtest::TEST_TOL));
    BOOST_LOG_TRIVIAL(info) << "\n***Min Auto inited passed ***\n";
    }

    delete [] res;
    return EXIT_SUCCESS;
}

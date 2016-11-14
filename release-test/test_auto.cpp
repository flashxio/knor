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

kpmbase::kmeans_t test_inited(double* p_centers, double* p_data,
        size_t* p_clust_asgn_cnt, unsigned* p_clust_asgns, const bool prune) {
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

    return ret;
}
} }

int main(int argc, char* argv[]) {
    std::vector<double> p_centers(kpmtest::TEST_K*kpmtest::TEST_NCOL);
    std::vector<double> p_data(kpmtest::TEST_NROW*kpmtest::TEST_NCOL);
    std::vector<size_t> p_clust_asgn_cnt(kpmtest::TEST_K);
    std::vector<unsigned> p_clust_asgns(kpmtest::TEST_NROW);

    kpmtest::init_log();

    {
        std::vector<double>res(kpmtest::TEST_K*kpmtest::TEST_NCOL);
        kpmtest::load_result(&res[0]);

        // Auto only
        {
            kpmbase::kmeans_t ret = kpmeans::test::test_inited(&p_centers[0],
                    &p_data[0], &p_clust_asgn_cnt[0], &p_clust_asgns[0], false);
            BOOST_VERIFY(kpmtest::check_collection_equal(
                        ret.centroids.begin(), ret.centroids.end(),
                        res.begin(), res.end(),
                        kpmtest::TEST_TOL));
            std::cout << "\n***Auto inited passed ***\n";
        }

        // Min auto
        {
            kpmbase::kmeans_t ret = kpmeans::test::test_inited(&p_centers[0],
                    &p_data[0], &p_clust_asgn_cnt[0], &p_clust_asgns[0], true);
            BOOST_VERIFY(kpmtest::check_collection_equal(
                        ret.centroids.begin(), ret.centroids.end(),
                        res.begin(), res.end(),
                        kpmtest::TEST_TOL));
            std::cout << "\n***Min Auto inited passed ***\n";
        }
    }
    return EXIT_SUCCESS;
}

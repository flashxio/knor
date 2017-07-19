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

#ifdef USE_NUMA
#include <numa.h>
#endif

#include "kmeans_coordinator.hpp"
#include "kmeans_task_coordinator.hpp"
#include "test_shared.hpp"
#include "util.hpp"

namespace kpmtest = kpmeans::test;
namespace kpmprune = kpmeans::prune;

namespace kpmeans { namespace test {
kpmbase::kmeans_t run_test(const std::string datafn, double* p_centers,
        size_t* p_clust_asgn_cnt, unsigned* p_clust_asgns, const bool prune,
        const std::string init, const unsigned max_iter) {
    constexpr unsigned NTHREADS = 2;
    unsigned nnodes = kpmbase::get_num_nodes();

    if (init == "none") {
            kpmbase::bin_io<double> br(TEST_INIT_CLUSTERS, TEST_K, TEST_NCOL);
            br.read(p_centers);
    }

    kpmbase::kmeans_t ret;
    if (prune) {
        kpmprune::kmeans_task_coordinator::ptr kc =
            kpmprune::kmeans_task_coordinator::create(
                datafn, TEST_NROW, TEST_NCOL, TEST_K, max_iter,
                nnodes, NTHREADS, p_centers, init, 0);
        ret = kc->run_kmeans();
    } else {
        kpmeans::kmeans_coordinator::ptr kc =
            kpmeans::kmeans_coordinator::create(datafn,
                TEST_NROW, TEST_NCOL, TEST_K, max_iter, nnodes,
                NTHREADS, p_centers, init, 0);
        ret = kc->run_kmeans();
    }
    return ret;
}
} }


int main(int argc, char* argv[]) {
    std::vector<double> p_centers(kpmtest::TEST_K*kpmtest::TEST_NCOL);
    std::vector<double> p_data(kpmtest::TEST_NROW*kpmtest::TEST_NCOL);
    std::vector<size_t> p_clust_asgn_cnt(kpmtest::TEST_K);
    std::vector<unsigned> p_clust_asgns(kpmtest::TEST_NROW);

    {
        std::vector<double>res(kpmtest::TEST_K*kpmtest::TEST_NCOL);
        kpmtest::load_result(&res[0]);

        /////////////////////////// Auto only ///////////////////////////
        {
            kpmbase::kmeans_t ret = kpmeans::test::run_test(
                    kpmtest::TESTDATA_FN, &p_centers[0],
                    &p_clust_asgn_cnt[0], &p_clust_asgns[0],
                    false, "none", 10);

            assert(kpmtest::check_collection_equal(
                        ret.centroids.begin(), ret.centroids.end(),
                        res.begin(), res.end(),
                        kpmtest::TEST_TOL));
            std::cout << "\n***Auto inited passed ***\n";
        }

        /////////////////////////// Min auto ///////////////////////////
        {
            kpmbase::kmeans_t ret = kpmeans::test::run_test(
                    kpmtest::TESTDATA_FN, &p_centers[0],
                    &p_clust_asgn_cnt[0], &p_clust_asgns[0],
                    false, "none", 10);
            assert(kpmtest::check_collection_equal(
                        ret.centroids.begin(), ret.centroids.end(),
                        res.begin(), res.end(),
                        kpmtest::TEST_TOL));
            std::cout << "\n***Min Auto inited passed ***\n";
        }

        //////////////////////////////////////////////////////////////////
        ////////////////////// Compare to each other /////////////////////
        //////////////////////////////////////////////////////////////////

        std::vector<std::string> inits;
        p_centers.clear();

        inits.push_back("random");
        inits.push_back("forgy");

        for (std::vector<std::string>::iterator it = inits.begin();
                it != inits.end(); ++it) {
            srand(1);
            kpmbase::kmeans_t ret_auto =
                kpmeans::test::run_test(
                    kpmtest::TESTDATA_FN, &p_centers[0],
                    &p_clust_asgn_cnt[0], &p_clust_asgns[0],
                    false, *it, 3);
            srand(1);
            kpmbase::kmeans_t ret_min_auto =
                kpmeans::test::run_test(
                    kpmtest::TESTDATA_FN, &p_centers[0],
                    &p_clust_asgn_cnt[0], &p_clust_asgns[0],
                    true, *it, 3);

            assert(std::equal(ret_auto.assignment_count.begin(),
                        ret_auto.assignment_count.end(),
                        ret_min_auto.assignment_count.begin()
                        ));
            assert(kpmtest::check_collection_equal(
                        ret_auto.centroids.begin(), ret_auto.centroids.end(),
                        ret_min_auto.centroids.begin(),
                        ret_min_auto.centroids.end(),
                        kpmtest::TEST_TOL));
        }
    }
    return EXIT_SUCCESS;
}

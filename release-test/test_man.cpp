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

#ifdef USE_NUMA
#include <numa.h>
#endif

#include "kmeans_coordinator.hpp"
#include "kmeans_task_coordinator.hpp"
#include "test_shared.hpp"
#include "util.hpp"

namespace ktest = knor::test;
namespace kprune = knor::prune;

namespace knor { namespace test {
kbase::kmeans_t run_test(const std::string datafn, double* p_centers,
        size_t* p_clust_asgn_cnt, unsigned* p_clust_asgns, const bool prune,
        const std::string init, const unsigned max_iter) {
    constexpr unsigned NTHREADS = 2;
    unsigned nnodes = kbase::get_num_nodes();

    if (init == "none") {
            kbase::bin_io<double> br(TEST_INIT_CLUSTERS, TEST_K, TEST_NCOL);
            br.read(p_centers);
    }

    kbase::kmeans_t ret;
    if (prune) {
        kprune::kmeans_task_coordinator::ptr kc =
            kprune::kmeans_task_coordinator::create(
                datafn, TEST_NROW, TEST_NCOL, TEST_K, max_iter,
                nnodes, NTHREADS, p_centers, init, 0);
        ret = kc->run();
    } else {
        knor::kmeans_coordinator::ptr kc =
            knor::kmeans_coordinator::create(datafn,
                TEST_NROW, TEST_NCOL, TEST_K, max_iter, nnodes,
                NTHREADS, p_centers, init, 0);
        ret = kc->run();
    }
    return ret;
}
} }


int main(int argc, char* argv[]) {
    std::vector<double> p_centers(ktest::TEST_K*ktest::TEST_NCOL);
    std::vector<double> p_data(ktest::TEST_NROW*ktest::TEST_NCOL);
    std::vector<size_t> p_clust_asgn_cnt(ktest::TEST_K);
    std::vector<unsigned> p_clust_asgns(ktest::TEST_NROW);

    {
        std::vector<double>res(ktest::TEST_K*ktest::TEST_NCOL);
        ktest::load_result(&res[0]);

        /////////////////////////// Auto only ///////////////////////////
        {
            kbase::kmeans_t ret = knor::test::run_test(
                    ktest::TESTDATA_FN, &p_centers[0],
                    &p_clust_asgn_cnt[0], &p_clust_asgns[0],
                    false, "none", 10);

            assert(ktest::check_collection_equal(
                        ret.centroids.begin(), ret.centroids.end(),
                        res.begin(), res.end(),
                        ktest::TEST_TOL));
            std::cout << "\n***Auto inited passed ***\n";
        }

        /////////////////////////// Min auto ///////////////////////////
        {
            kbase::kmeans_t ret = knor::test::run_test(
                    ktest::TESTDATA_FN, &p_centers[0],
                    &p_clust_asgn_cnt[0], &p_clust_asgns[0],
                    false, "none", 10);
            assert(ktest::check_collection_equal(
                        ret.centroids.begin(), ret.centroids.end(),
                        res.begin(), res.end(),
                        ktest::TEST_TOL));
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
            kbase::kmeans_t ret_auto =
                knor::test::run_test(
                    ktest::TESTDATA_FN, &p_centers[0],
                    &p_clust_asgn_cnt[0], &p_clust_asgns[0],
                    false, *it, 3);
            srand(1);
            kbase::kmeans_t ret_min_auto =
                knor::test::run_test(
                    ktest::TESTDATA_FN, &p_centers[0],
                    &p_clust_asgn_cnt[0], &p_clust_asgns[0],
                    true, *it, 3);

            assert(std::equal(ret_auto.assignment_count.begin(),
                        ret_auto.assignment_count.end(),
                        ret_min_auto.assignment_count.begin()
                        ));
            assert(ktest::check_collection_equal(
                        ret_auto.centroids.begin(), ret_auto.centroids.end(),
                        ret_min_auto.centroids.begin(),
                        ret_min_auto.centroids.end(),
                        ktest::TEST_TOL));
        }
    }
    return EXIT_SUCCESS;
}
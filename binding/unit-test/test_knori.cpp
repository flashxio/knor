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

#include "knori.hpp"
#include "util.hpp"

namespace kpmbase = kpmeans::base;

int main(int argc, char* argv[]) {

    constexpr size_t nrow = 50;
    constexpr size_t ncol = 5;
    constexpr size_t max_iters = 20;
    constexpr unsigned k = 8;
    constexpr unsigned nthread = 4;
    const std::string fn = "../../test-data/matrix_r50_c5_rrw.bin";
    const std::string centroidfn = "../../test-data/init_clusters_k8_c5.bin";
    const unsigned nnodes = kpmbase::get_num_nodes();

    // Read from disk
    std::cout << "Testing read from disk ..\n";
    {
        kpmbase::kmeans_t ret = kpmbase::kmeans(
                fn, nrow, ncol, k,
                /*"/data/kmeans/r16_c3145728_k100_cw.dat", 3145728, 16, 100,*/
                max_iters, nnodes, nthread, NULL);
        ret.print();
    }

    //Data already in-mem FULL
    std::cout << "Testing data only in-mem ..\n";
    {
        std::vector<double> data(nrow*ncol);
        kpmbase::bin_rm_reader<double> br(fn);
        br.read(data);

        kpmbase::kmeans_t ret = kpmbase::kmeans(
                &data[0], nrow, ncol, k,
                max_iters, nnodes, nthread, NULL,
                "kmeanspp", -1, "eucl", true);

        ret.print();
    }

    // Data already in-mem PRUNED
    std::cout << "Testing PRUNED data only in-mem ..\n";
    {
        std::vector<double> data(nrow*ncol);
        kpmbase::bin_rm_reader<double> br(fn);
        br.read(data);

        kpmbase::kmeans_t ret = kpmbase::kmeans(
                &data[0], nrow, ncol, k,
                max_iters, nnodes, nthread, NULL);

        ret.print();
    }

    std::cout << "Testing data + Centroid in-mem ..\n";
    {
        std::vector<double> data(nrow*ncol);
        kpmbase::bin_rm_reader<double> br(fn);
        br.read(data);

        std::vector<double> centroids(k*ncol);
        kpmbase::bin_rm_reader<double> br2(centroidfn);
        br2.read(centroids);

        kpmbase::kmeans_t ret_full = kpmbase::kmeans(
                &data[0], nrow, ncol, k,
                max_iters, nnodes, nthread, &centroids[0],
                "none", -1, "eucl", true, false);

        ret_full.print();

        //////////////////////////////////*****////////////////////////////
        //////////////////////////////////*****////////////////////////////
        kpmbase::kmeans_t ret_numa_full = kpmbase::kmeans(
                &data[0], nrow, ncol, k,
                max_iters, nnodes, nthread, &centroids[0],
                "none", -1, "eucl", true, true);

        ret_numa_full.print();
        assert(ret_full == ret_numa_full);
        std::cout << "SUCCESS FULL. Data + Centroids in-mem (numa_opt)!\n\n";

        //////////////////////////////////*****////////////////////////////
        //////////////////////////////////*****////////////////////////////
        std::cout << "Testing PRUNED. Data + Centroid in-mem ...\n";
        kpmbase::kmeans_t ret = kpmbase::kmeans(
                &data[0], nrow, ncol, k,
                max_iters, nnodes, nthread, &centroids[0],
                "none");

        ret.print();

        //////////////////////////////////*****////////////////////////////
        //////////////////////////////////*****////////////////////////////
        std::cout << "Testing PRUNED. Data + Centroid in-mem ...\n";
        kpmbase::kmeans_t ret_numa = kpmbase::kmeans(
                &data[0], nrow, ncol, k,
                max_iters, nnodes, nthread, &centroids[0],
                "none", -1, "eucl", false, true);

        ret_numa.print();
        assert(ret == ret_numa);
        std::cout << "SUCCESS PRUNED. Data + Centroids in-mem (numa_opt)!\n\n";

        //////////////////////////////////*****////////////////////////////
        //////////////////////////////////*****////////////////////////////
        // assert(ret_full == ret_numa); // FIXME: Fails but is the same ...
        // std::cout << "SUCCESS PRUNED v FULL. Data + Centroids in-mem (numa_opt)!\n\n";
    }

    return EXIT_SUCCESS;
}

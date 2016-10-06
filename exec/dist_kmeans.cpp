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

#include "dist_kmeans.hpp"

int main(int argc, char* argv[]) {
    El::Environment env(argc, argv);

    try {
        const std::string data_fn =
            El::Input<std::string>("-f","datafile (TSV)","");
        const El::Unsigned k = El::Input("-k","number of clusters",2);
        const size_t nsamples =
            El::Input("-n","number of samples in data",10);
        const size_t dim =
            El::Input("-d","Dimensionality of data",4);
        std::string init = El::Input<std::string>("-I",
                "initialization method [forgy | random | plusplus]","forgy");
        const El::Unsigned max_iters =
            El::Input("-i","number of iterations",10);
        const El::Int seed = El::Input("-s","seeding for random init",2);
        const double tol = El::Input("-T","Convergence tolerance",1E-6);
        const std::string centroid_fn =
            El::Input<std::string>("-c","Pre-initialized centroids","");
        const bool prune = El::Input("-p","Use triangle inequality", false);
        El::ProcessInput();

        El::mpi::Comm comm = El::mpi::COMM_WORLD;
        El::Grid grid(comm);
        El::Unsigned rank = El::mpi::Rank(comm);

        El::DistMatrix<double, El::STAR, El::VC> data(dim, nsamples);
        El::Matrix<double> centroids;

        if (!data_fn.empty()) {
            //El::Read(data, data_fn, El::ASCII);
            El::Read(data, data_fn, El::BINARY_FLAT);
            if (rank == root) {
                El::Output("Read complete for proc: ", rank);
                El::Output("Dim: (", data.Height(), ", ", data.Width() ,")");
            }
#if KM_DEBUG
            El::Print(data, "Data:");
#endif
        } else {
            El::Output("Creating random data:");
            El::Uniform(data, dim, nsamples);
        }

        if (!centroid_fn.empty()) {
            init = "none";
            El::Read(centroids, centroid_fn, El::ASCII);
        } else {
            El::Zeros(centroids, data.Height(), k);
        }

        // Do some checking
        assert(data.Height() == centroids.Height());
        assert(k == (El::Unsigned)centroids.Width());

        if (rank == root) El::Output("Starting k-means ...");

        if (prune) {
            if (rank == root) El::Output("Running pruned ...");
            kpmeans::run_tri_kmeans<double>(data, centroids, k,
                    tol, init, seed, max_iters);
        } else {
            if (rank == root) El::Output("Running full ...");
            kpmeans::run_kmeans<double>(data, centroids, k,
                    tol, init, seed, max_iters);
        }
    }
    catch(std::exception& e) { El::ReportException(e); }

    return EXIT_SUCCESS;
}

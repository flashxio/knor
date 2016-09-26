#include <El.hpp>
#include "kmeans.hpp"

/*******************************************/
namespace skyml = skylark::ml;
/*******************************************/

int main(int argc, char* argv[]) {
    El::Environment env(argc, argv);

    try {
        const std::string data_fn =
            El::Input<std::string>("--f","datafile (TSV)","");
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
        El::ProcessInput();

        El::mpi::Comm comm = El::mpi::COMM_WORLD;
        El::Grid grid(comm);
        El::Unsigned rank = El::mpi::Rank(comm);

        El::Matrix<double> centroids(dim, k);
        El::Zero(centroids);
        El::DistMatrix<double, El::STAR, El::VC> data;

        if (!data_fn.empty()) {
            El::Read(data, data_fn, El::ASCII);
#if KM_DEBUG
            if (rank == root) El::Output("Read complete for proc: ", rank);
            El::Print(data, "Data:");
#endif
        }
        else {
            El::Output("Creating random data:");
            El::Uniform(data, dim, nsamples);
        }

        if (!centroid_fn.empty()) {
            init = "none";
            El::Read(centroids, centroid_fn, El::ASCII);
        }

        if (rank == root) El::Output("Starting k-means ...");
        skyml::run_kmeans<double>(data, centroids, k,
                tol, init, seed, max_iters);
    }
    catch(std::exception& e) { El::ReportException(e); }

    return EXIT_SUCCESS;
}

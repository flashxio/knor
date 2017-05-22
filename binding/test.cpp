#include "knori.hpp"

int main(int argc, char* argv[]) {

    constexpr size_t nrow = 50;
    constexpr size_t ncol = 5;
    constexpr size_t max_iters = 20;
    constexpr unsigned k = 8;
    constexpr unsigned nthread = 2;
    const std::string fn = "../test-data/matrix_r50_c5_rrw.bin";

    // Read from disk
    std::cout << "Testing read from disk ..\n";
    {
    kpmbase::kmeans_t ret = kpmbase::kmeans(
            fn, nrow, ncol, k,
            /*"/data/kmeans/r16_c3145728_k100_cw.dat", 3145728, 16, 100,*/
            max_iters, numa_num_task_nodes(), nthread, NULL);
    ret.print();
    }

    // Data already in-mem FULL
    std::cout << "Testing data only in-mem ..\n";
    {
    std::vector<double> data(nrow*ncol);
    kpmbase::bin_rm_reader<double> br(fn);
    br.read(data);

    kpmbase::kmeans_t ret = kpmbase::kmeans(
            &data[0], nrow, ncol, k,
            max_iters, numa_num_task_nodes(), nthread, NULL,
            "kmeanspp", -1, "eucl", "", true);

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
            max_iters, numa_num_task_nodes(), nthread, NULL);

    ret.print();
    }
    return EXIT_SUCCESS;
}

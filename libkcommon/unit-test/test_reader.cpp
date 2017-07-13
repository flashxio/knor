/**
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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "io.hpp"
#include "util.hpp"

namespace kpmbase = kpmeans::base;

void test_text_reader(std::string fn, const size_t NROW, const size_t NCOL) {
    std::cout << "Text reader test ...\n";
    kpmbase::text_reader<double> rdr(fn);
    rdr.set_ncol(NCOL);
    std::vector<double> v(NCOL);
    std::vector<double> m;

    while (rdr.readline(v))
        m.insert(m.end(), v.begin(), v.end());

    std::cout << "Readline: ==>\n";
    std::cout << "nrow: " << rdr.get_nrow() <<
        ", ncol: " << rdr.get_ncol() << std::endl;
    kpmbase::print_mat<double>(&m[0], rdr.get_nrow(), rdr.get_ncol());

    std::cout << "Read: ==>\n";
    std::vector<double> v2(NROW*NCOL);
    kpmbase::text_reader<double> rdr2(fn);
    rdr2.set_ncol(NCOL);
    rdr2.read(v2);
    std::cout << "nrow: " << rdr2.get_nrow() <<
        ", ncol: " << rdr2.get_ncol() << std::endl;

    kpmbase::print_mat<double>(&v2[0], rdr2.get_nrow(), rdr2.get_ncol());

    assert(m.size() == NROW * NCOL);
    assert(m.size() == v2.size());
    assert(kpmbase::eq_all<double>(&m[0], &v2[0], m.size()));
}

void test_bin_rm_reader(std::string fn, const size_t NROW, const size_t NCOL) {
    std::cout << "\nBinary reader test ...\n";

    kpmbase::bin_rm_reader<double> rdr(fn);
    rdr.set_ncol(NCOL);
    std::vector<double> v(NCOL);

    while (rdr.readline(v)) {
        std::cout << "line: ==> ";
        kpmbase::print_vector<double>(v);
    }
}

int main(int argc, char* argv[]) {
    size_t nrow = 5;
    size_t ncol = 3;
    test_text_reader("test.txt", nrow, ncol);
    test_bin_rm_reader("test.dat", nrow, ncol);

    return EXIT_SUCCESS;
}

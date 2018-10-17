/**
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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>

#include "dense_matrix.hpp"
#include "linalg.hpp"

using namespace knor::base;

int main() {
    Eigen::MatrixXd m(3,3);
    m << 3, 2, 3,
          4, 5, 6,
          7, 8, .5;
    std::cout << m << std::endl;
    std::cout << "m.adjoint():\n" << m.adjoint() << std::endl;

    Eigen::MatrixXd centered = m.colwise() - m.rowwise().mean();
    std::cout<< "\ncentered:\n" << centered << "\n";
    std::cout << "\nmeans:\n" << m.rowwise().mean() << "\n";
    std::cout << "\n centered.adjoint():\n" << centered.adjoint() << "\n";

    Eigen::MatrixXd cov = (centered * centered.adjoint()) / double(m.rows() - 1);
    std::cout<< "\ncov: \n" << cov << "\n";

    ////////////////////////////// Test our sh*t ///////////////////////////////
    dense_matrix<double>::rawptr dm = dense_matrix<double>::create(3,3);
    double rd[9] = {3,2,3,4,5,6,7,8,.5};
    dm->set(rd);
    std::cout << "\n\n\nTesting dense matrix: \n";
    dm->print();

    for (size_t row = 0; row < dm->get_nrow(); row++)
        for (size_t col = 0; col < dm->get_ncol(); col++)
            assert(dm->get(row,col) == m(row, col));
    std::cout << "Eigen matrix = dense matrix\n";

    std::vector<double> dmmean;
    dm->mean(dmmean, 0);

    std::cout << "Column means are: \n";
    print_vector(dmmean);
    dense_matrix<double>* dmcentered = (*dm) - dmmean;

    std::cout << "Centered:\n";
    dmcentered->print();

#if 0 // FIXME: Bug here
    auto adj = linalg::adjoint(dmcentered);
#else
    double _[9] = {3,4,7,2,5,8,3,6,.5};
    auto adj = dense_matrix<double>::create(3,3);
    adj->set(_);
#endif
    std::cout << "Centered adjoint:\n";
    adj->print();

    auto dcov = (*dmcentered) * (*adj);
    (*dcov) /= (dm->get_nrow()-1);
    std::cout << "cov: \n";
    dcov->print();

    // Cleanup
    delete dm;
    delete dmcentered;
    delete adj;
    delete dcov;
    return EXIT_SUCCESS;
}

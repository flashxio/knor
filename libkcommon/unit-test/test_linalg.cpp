
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

// C++ program to find adjoint and inverse of a matrix

#include "linalg.hpp"
#include "io.hpp"
#include "dense_matrix.hpp"

namespace kbase = knor::base;
constexpr unsigned N = 6;

void test_det() {
	double A[N*N] = { 5, -2, 2, 7, 12, 2,
					1, 0, 0, 3, 1, 2,
					-3, 1, 5, 0, 21, 2,
					3, -1, -9, 4, 1, 3,
					13, -12, -9, 4, 1, 4,
					12, -12, -9, 4, 1, 6};

	double adj[N*N]; // To store adjoint of A*
	double inv[N*N]; // To store inverse of A*

    printf("Input matrix is :\n");
    kbase::print(A, N, N);

    printf("\nThe Determinant is : %.2f\n",
            kbase::linalg::determinant(A,N, N));

    printf("\nThe Inverse is :\n");
	if (kbase::linalg::inverse(A, inv, N))
		kbase::print(inv, N, N);
}

void test_det_dense_mat() {
    kbase::dense_matrix<double>::rawptr dm
        = kbase::dense_matrix<double>::create(N, N);

	double A[N*N] = { 5, -2, 2, 7, 12, 2,
					1, 0, 0, 3, 1, 2,
					-3, 1, 5, 0, 21, 2,
					3, -1, -9, 4, 1, 3,
					13, -12, -9, 4, 1, 4,
					12, -12, -9, 4, 1, 6 };
    dm->set(A);

    kbase::dense_matrix<double>::rawptr inv
        = kbase::dense_matrix<double>::create(N, N);

    assert(dm->get_nrow() == dm->get_ncol());
    assert(dm->get_nrow() == inv->get_nrow() &&
            dm->get_ncol() == inv->get_ncol());
    printf("Input (dense) matrix is :\n");
    dm->print();

    printf("\nThe Determinant is : %.2f\n",
            kbase::linalg::determinant(dm->as_pointer(),
                dm->get_nrow(), dm->get_ncol()));

    printf("\nThe Inverse is :\n");
	if (kbase::linalg::inverse(dm->as_pointer(), inv->as_pointer(),
                dm->get_nrow()))
        inv->print();

    delete dm;
    delete inv;
}

int main() {
    test_det();
    test_det_dense_mat();

    printf("Successful 'testit' test ...\n");
    return EXIT_SUCCESS;
}
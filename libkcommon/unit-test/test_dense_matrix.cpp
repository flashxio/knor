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

#include <stdio.h>
#include <iostream>

#include <cassert>
#include "dense_matrix.hpp"

using namespace knor::base;

void setit(const size_t NROW,  const size_t NCOL,
        dense_matrix<double>::rawptr mat) {
    for (size_t row = 0; row < NROW; row++) {
        for (size_t col = 0; col < NCOL; col++) {
            mat->set(row, col, row*NCOL+col);
        }
    }
}

void checkit(const size_t NROW,  const size_t NCOL,
        dense_matrix<double>::rawptr mat) {
    for (size_t row = 0; row < NROW; row++) {
        for (size_t col = 0; col < NCOL; col++) {
           assert(mat->get(row, col) == row*NCOL+col);
        }
    }
}

void test_dense_matrix() {
    {
        size_t NROW = 16;
        size_t NCOL = 4;

        dense_matrix<double>::rawptr mat =
            dense_matrix<double>::create(NROW, NCOL);
        setit(NROW, NCOL, mat);
        checkit(NROW, NCOL, mat);
        delete mat;
    }

    {
        size_t NROW = 32;
        size_t NCOL = 16;
        dense_matrix<double>::rawptr mat =
            dense_matrix<double>::create(NROW, NCOL);
        mat->set_nrow(NROW);
        mat->set_ncol(NCOL);

        setit(NROW, NCOL, mat);
        checkit(NROW, NCOL, mat);
        delete mat;
    }
}

void test_dot() {
    std::vector<double> lv = {1, 2, 3, 4};
    std::vector<double> rv = {1, 2, 3, 5, 6, 7};

    dense_matrix<double>::rawptr l = dense_matrix<double>::create(2,2);
    l->set(&lv[0]);
    dense_matrix<double>::rawptr r = dense_matrix<double>::create(2,3);
    r->set(&rv[0]);

    dense_matrix<double>::rawptr res = (*l)*(*r);

    dense_matrix<double>::rawptr man_res = dense_matrix<double>::create(2,3);
    std::vector<double> man_resv = {11, 14, 17, 23, 30, 37};

    std::cout << "Manual result: \n";
    print_mat(&man_resv[0], 2, 3);

    std::cout << "Computed result:\n";
    res->print();

    man_res->set(&man_resv[0]);
    assert((*res) == (*man_res));

    delete (l);
    delete (r);
    delete (res);
    delete (man_res);
}

int main() {
    test_dense_matrix();
    printf("Successful 'test_dense_matrix' test ...\n");

    test_dot();
    printf("Successful 'test_dot' for dense_matrix test ...\n");
    return EXIT_SUCCESS;
}

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

int main() {
    test_dense_matrix();
    printf("Successful 'test_dense_matrix' test ...\n");
    return EXIT_SUCCESS;
}

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
#include "io.hpp"

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

void test_basic() {
    std::vector<double> lv = {1, 2, 3, 5, 6, 7};
    dense_matrix<double>::rawptr l = dense_matrix<double>::create(2,3);
    l->set(&lv[0]);

    dense_matrix<double>* tmp = dense_matrix<double>::create();
    tmp->copy_from(l);

    // Make sure attributes are copied over
    assert(tmp->get_nrow() == l->get_nrow());
    assert(tmp->get_ncol() == l->get_ncol());

    printf("tmp:\n"); tmp->print();
    printf("l:\n"); l->print();

    // Make sure the data is copied over
    for (size_t row = 0; row < l->get_nrow(); row++)
        for (size_t col = 0; col < l->get_ncol(); col++)
            assert(tmp->get(row, col) == l->get(row, col));

    // Make sure it's not a reference and is a deep copy
    l->set(0,0,0);
    assert(tmp->get(0,0) != l->get(0,0));
    tmp->set(0,0,0);

    ///////////////////////////////// Test Mean ////////////////////////////////
    std::vector<double> mean;
    l->mean(mean, 1);

    assert(mean.size() == 3);
    assert(mean[0] == 2.5);
    assert(mean[1] == 4);
    assert(mean[2] == 5);

    tmp->mean(mean,0);
    assert(mean.size() == 2);
    assert(mean[0] == (5/((double)3)));
    assert(mean[1] == 6);
    ////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// Test Sub /////////////////////////////////

    auto sub = (*tmp) - (*l);
    assert(sub->get_nrow() == 2 && sub->get_ncol() == 3);

    for (size_t row = 0; row < sub->get_nrow(); row++)
        for (size_t col = 0; col < sub->get_ncol(); col++)
            assert(sub->get(row, col) == 0);

    ////////////////////////////////////////////////////////////////////////////

    *tmp += *l;
    printf("Sum: \n"); tmp->print();

    *tmp += *tmp;
    printf("Sum: \n"); tmp->print();

    delete (tmp);
    delete (l);
}

int main() {
    test_dense_matrix();
    printf("Successful 'test_dense_matrix' test ...\n");

    test_dot();
    printf("Successful 'test_dot' for dense_matrix test ...\n");

    test_basic();
    printf("Successful 'test_basic' for dense_matrix test ...\n");
    return EXIT_SUCCESS;
}

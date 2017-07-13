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

#include <stdio.h>
#include <iostream>

#include <cassert>

#include "clusters.hpp"

namespace kpmbase = kpmeans::base;

constexpr unsigned NCOL = 4;
constexpr unsigned NCLUST = 5;
static const kpmbase::kmsvector zero {1,2,3,4};
static const kpmbase::kmsvector one {4,5,6,7,8};
static const kpmbase::kmsvector two {9,10,11,12};
static const kpmbase::kmsvector three {13,14,15,16};
static const kpmbase::kmsvector four {17,18,19,20};
static const kpmbase::clusters::ptr empty =
        kpmbase::clusters::create(NCLUST, NCOL);
static const std::vector<kpmbase::kmsvector> data {zero, one, two, three, four};
static const double arr [] =
    {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
static const kpmbase::kmsvector kv(std::begin(arr), std::end(arr));

void test_clusters() {
    printf("Testing clusters ...\n");
    printf("Print no init:\n");
    empty->print_means();

    kpmbase::clusters::ptr cls = kpmbase::clusters::create(NCLUST, NCOL, kv);
    printf("Print after init:\n");
    cls->print_means();

    cls->clear();
    printf("After clear test:\n");
    assert(*cls == *empty);
    printf("Success ...\n");

    cls->set_mean(kv);
    printf("Print after all set-mean:\n");
    cls->print_means();

    cls->set_mean(zero, 2); cls->set_mean(zero, 4);
    printf("Print after all setting 0 to 2 & 4 :\n");
    cls->print_means();

    printf("Clearing all means & adding all data to all clusters:\n");
    cls->clear();
    assert(*cls == *empty);
    printf("Success ...\n");

    for (unsigned cl = 0; cl < NCLUST; cl++)
        for (unsigned i = 0; i < data.size(); i++)
            cls->add_member(&(data[i][0]), cl);
    printf("Before finalize all should be equal:\n");
    kpmbase::clusters::ptr old = kpmbase::clusters::create(NCLUST, NCOL);
    *old = *cls;

    cls->print_means();

    cls->finalize_all();
    printf("After finalize all should be equal as well:\n");
    cls->print_means();

    printf("Test unfinalize should return to original:\n");
    cls->unfinalize_all();
    assert(*old == *cls);
    printf("Success ...\n");

    printf("Removing members:\n");
    for (unsigned cl = 0; cl < NCLUST; cl++)
        for (unsigned i = 0; i < data.size(); i++)
            cls->remove_member(&(data[i][0]), cl);

    cls->print_means();
    assert(*cls == *empty);
    printf("Success ...\n");

    cls->peq(empty);
    assert(*cls == *empty);
    printf("Success ...\n");
}

void test_prune_clusters() {
    printf("Testing prune_clusters ...\n");
    kpmbase::prune_clusters::ptr pcl =
        kpmbase::prune_clusters::create(NCLUST, NCOL);
    pcl->set_mean(arr);

    printf("Testing set_mean == create(nclust, ncol, kv)");
    assert(*pcl == *(kpmbase::prune_clusters::create(NCLUST, NCOL, kv)));
    printf("Success ...\n");

    kpmbase::clusters::ptr cl = kpmbase::clusters::create(NCLUST, NCOL, kv);
    printf("Test *cl == *pcl after cast...\n");
    assert(*cl == *(std::static_pointer_cast<kpmbase::clusters,
                kpmbase::prune_clusters>(pcl)));
    pcl->clear();

    printf("Setting test cl to values == pcl ...\n");
    for (unsigned i = 0; i < NCLUST; i++) {
        pcl->add_member(&(data[i][0]), i);
        pcl->finalize(i);
    }

    pcl->set_prev_means();
    printf("Printing prev_mean:\n");
    pcl->print_prev_means_v();

    printf("Testing prev_mean ..\n");

    for (unsigned i = 0; i < NCLUST; i++) {
        kpmbase::kmsvector v = pcl->get_prev_means();
        kpmbase::kmsiterator it = v.begin() + (i*NCOL);
        for (unsigned col = 0; col < NCOL; col++) {
            assert(*(it++) == data[i][col]);
        }
    }
    printf("Success ...\n");
}

int main() {
    test_clusters();
    test_prune_clusters();
    return EXIT_SUCCESS;
}

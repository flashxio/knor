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

#include "clusters.hpp"

namespace kbase = knor::core;

constexpr unsigned NCOL = 4;
constexpr unsigned NCLUST = 5;
static const kbase::kmsvector zero {1,2,3,4};
static const kbase::kmsvector one {4,5,6,7,8};
static const kbase::kmsvector two {9,10,11,12};
static const kbase::kmsvector three {13,14,15,16};
static const kbase::kmsvector four {17,18,19,20};
static const kbase::clusters::ptr empty =
        kbase::clusters::create(NCLUST, NCOL);
static const std::vector<kbase::kmsvector> data {zero, one, two, three, four};
static const double arr [] =
    {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
static const kbase::kmsvector kv(std::begin(arr), std::end(arr));

void test_clusters() {
    printf("Testing clusters ...\n");
    printf("Print no init:\n");
    empty->print_means();

    kbase::clusters::ptr cls = kbase::clusters::create(NCLUST, NCOL, kv);
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
    kbase::clusters::ptr old = kbase::clusters::create(NCLUST, NCOL);
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

void test_scaling() {
    kbase::clusters::ptr cls = kbase::clusters::create(NCLUST, NCOL, kv);
    cls->print_means();

    for (unsigned i = 0; i < NCLUST; i++) {
        cls->scale_centroid(2, i, &kv[NCOL*i]);
    }

    kbase::clusters::ptr cls2 = kbase::clusters::create(NCLUST, NCOL, kv);
    assert(*cls2 == *cls);
}

void test_prune_clusters() {
    printf("Testing prune_clusters ...\n");
    kbase::prune_clusters::ptr pcl =
        kbase::prune_clusters::create(NCLUST, NCOL);
    pcl->set_mean(arr);

    printf("Testing set_mean == create(nclust, ncol, kv)");
    assert(*pcl == *(kbase::prune_clusters::create(NCLUST, NCOL, kv)));
    printf("Success ...\n");

    kbase::clusters::ptr cl = kbase::clusters::create(NCLUST, NCOL, kv);
    printf("Test *cl == *pcl after cast...\n");
    assert(*cl == *(std::static_pointer_cast<kbase::clusters,
                kbase::prune_clusters>(pcl)));
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
        kbase::kmsvector v = pcl->get_prev_means();
        kbase::kmsiterator it = v.begin() + (i*NCOL);
        for (unsigned col = 0; col < NCOL; col++) {
            assert(*(it++) == data[i][col]);
        }
    }
    printf("Success ...\n");
}

void test_hclusters() {
    printf("Testing hcluster ...\n");
    auto hclust0 = kbase::h_clusters::create(2, NCOL);
    hclust0->set_mean(zero, 0);
    hclust0->set_mean(one, 1);
    hclust0->print_means();

    kbase::h_clusters::cast2(hclust0)->set_zeroid(0);
    kbase::h_clusters::cast2(hclust0)->set_oneid(1);

    auto hclust1 = kbase::h_clusters::create(2, NCOL, arr);
    hclust1->print_means();
    printf("Success ...\n");
}

int main() {
    test_clusters();
    test_prune_clusters();
    test_scaling();
    test_hclusters();
    return EXIT_SUCCESS;
}

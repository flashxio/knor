/*
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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY CURRENT_KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "util.hpp"
#include <cassert>

using namespace knor::base;

void test_hclust_floor() {
    assert(get_hclust_floor(1) == 1);
    assert(get_hclust_floor(2) == 2);
    assert(get_hclust_floor(3) == 2);
    assert(get_hclust_floor(4) == 4);
    assert(get_hclust_floor(5) == 4);
    assert(get_hclust_floor(6) == 4);
    assert(get_hclust_floor(7) == 4);
    assert(get_hclust_floor(8) == 8);
    assert(get_hclust_floor(9) == 8);
    assert(get_hclust_floor(10) == 8);
    assert(get_hclust_floor(15) == 8);
    assert(get_hclust_floor(16) == 16);
    assert(get_hclust_floor(20) == 16);
    assert(get_hclust_floor(27) == 16);
    assert(get_hclust_floor(31) == 16);
    assert(get_hclust_floor(32) == 32);
    printf("hclust_floor test OK ...\n");
}

void test_hclust_ceil() {
    assert(get_hclust_ceil(1) == 1);
    assert(get_hclust_ceil(2) == 2);
    assert(get_hclust_ceil(3) == 4);
    assert(get_hclust_ceil(4) == 4);
    assert(get_hclust_ceil(5) == 8);
    assert(get_hclust_ceil(6) == 8);
    assert(get_hclust_ceil(7) == 8);
    assert(get_hclust_ceil(8) == 8);
    assert(get_hclust_ceil(9) == 16);
    assert(get_hclust_ceil(10) == 16);
    assert(get_hclust_ceil(15) == 16);
    assert(get_hclust_ceil(16) == 16);
    assert(get_hclust_ceil(20) == 32);
    assert(get_hclust_ceil(27) == 32);
    assert(get_hclust_ceil(31) == 32);
    assert(get_hclust_ceil(32) == 32);
    assert(get_hclust_ceil(33) == 64);
    assert(get_hclust_ceil(63) == 64);

    printf("hclust_ceil test OK ...\n");
}

int main() {
    test_hclust_floor();
    test_hclust_ceil();
    printf("Successful util test!\n");
}

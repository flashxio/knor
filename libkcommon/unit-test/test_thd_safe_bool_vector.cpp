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

#include <stdlib.h>
#include <iostream>
#include <cassert>

#include "thd_safe_bool_vector.hpp"
#include "util.hpp"

namespace kpmbase = kpmeans::base;

void build_state(std::vector<short> &verifier,
        kpmbase::thd_safe_bool_vector::ptr data, const unsigned len) {
    for (unsigned i = 0; i < len; i++) {
        short rand_val = rand() % 2;
        if (rand_val == 0) {
            verifier[i] = rand_val;
            data->set(i, false);
        } else if (rand_val == 1) {
            verifier[i] = rand_val;
            data->set(i, true);
        } else {
            assert(0);
        }
    }
    printf("Built state successfully ...\n");
}

template <typename T>
void test_correctness(const std::vector<T> &verifier,
        const kpmbase::thd_safe_bool_vector::ptr data) {
    assert(verifier.size() == data->size());

    for (unsigned i = 0; i < verifier.size(); i++)
        assert((bool)verifier[i] == data->get(i));
}

void test_init_ctor(const unsigned len) {
    kpmbase::thd_safe_bool_vector::ptr data =
        kpmbase::thd_safe_bool_vector::create(len, true);
    for (unsigned i = 0; i < len; i++)
        assert(data->get(i) == true);

    data = kpmbase::thd_safe_bool_vector::create(len, false);
    for (unsigned i = 0; i < len; i++)
        assert(data->get(i) == false);

    printf("Successfully init ctor ...\n");
}

void test_thread_safety(const kpmbase::thd_safe_bool_vector::ptr data,
        const unsigned nthreads) {

    constexpr unsigned NUM_TESTS = 50;
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        // Make the _test vector
        std::vector<bool> _test;
        for (unsigned i = 0; i < data->size(); i++) {
            short rand_val = rand() % 2;
            if (rand_val)
                _test.push_back(true);
            else
                _test.push_back(false);
        }

        // Assign _test values to data
#ifdef _OPENMP
#pragma omp parallel for shared(_test)
#endif
        for (unsigned i = 0; i < data->size(); i++) {
            data->set(i, _test[i]);
        }
        test_correctness<bool>(_test, data);
    }
    printf("Successfully passed thread safety ...\n");
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        fprintf(stderr, "usage: ./test_thd_safe_bool_vector nthreads len\n");
        exit(EXIT_FAILURE);
    }

    unsigned nthreads = atol(argv[1]);
    unsigned len = atol(argv[2]);

    kpmbase::thd_safe_bool_vector::ptr data =
        kpmbase::thd_safe_bool_vector::create(len);
    std::vector<short> verifier(len);

    build_state(verifier, data, len);
    test_correctness<short>(verifier, data);
    printf("Successfully passed correctness ...\n");
    test_init_ctor(len);

    test_thread_safety(data, nthreads);

    return (EXIT_SUCCESS);
}

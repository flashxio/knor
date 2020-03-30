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
#include <memory>
#include <random>
#include "types.hpp"

using namespace knor;

class Test {
    private:
        Test(int val) {
            this->val = val;
        }
    public:
        typedef std::shared_ptr<Test> ptr;

        int val;
        static ptr create(int val) {
            return ptr(new Test(val));
        }

        void print() {
            std::cout << "val: " << val << std::endl;
        }

        std::string str() {
            return std::to_string(val);
        }
};

void test_vector_map() {
    unsigned cap = 8;
    //core::vmap<Test::ptr> vm(cap, nullptr);
    core::vmap<Test::ptr> vm;
    vm.set_capacity(cap);

    for (size_t i = 0; i < vm.size(); i++)
        vm[i] = Test::create(i);

    auto itr = vm.get_iterator();
    while (itr.has_next())
        itr.next().second->print();

    printf("Deleting 0, 3 & 5\n");
    vm.erase(0);
    vm.erase(3);
    vm.erase(5);

    auto itr2 = vm.get_iterator();
    while (itr2.has_next())
        itr2.next().second->print();

#if 1
    printf("Access a few:\n");
    printf("2 is --> %s\n", vm[2]->str().c_str());
    printf("1 is --> %s\n", vm[1]->str().c_str());
    printf("6 is --> %s\n", vm[6]->str().c_str());
#endif

    std::vector<size_t> tmp{1,2,4,6,7};
    std::vector<size_t> ids;
    vm.get_keys(ids);

    assert(tmp.size() == ids.size());

    for (size_t i = 0; i < ids.size(); i++) {
        assert(ids[i] == tmp[i]);
        assert(vm.has_key(ids[i]));
    }

    assert(!vm.has_key(0));
    assert(!vm.has_key(3));
    assert(!vm.has_key(5));
    assert(!vm.has_key(8));
    printf("Successful 'test_v_map' test ...\n");
}

int main() {
    test_vector_map();
    return EXIT_SUCCESS;
}

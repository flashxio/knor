/*
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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY CURRENT_KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string.h>
#include <limits>
#include "io.hpp"
#include "thd_safe_bool_vector.hpp"

namespace kpmeans { namespace base {

_bool::_bool(char c) {
    _[0] = c;
    _[1] = 0;
}

_bool::operator bool() const {
if (strcmp(_, "1"))
    return false;
return true;
}

const bool thd_safe_bool_vector::get(const unsigned idx) const {
    return data[idx];
}

void thd_safe_bool_vector::set(const unsigned idx, const bool val) {
    if (val) {
        data[idx] = _bool('1');
    } else {
        data[idx] = _bool('0');
    }
}

unsigned thd_safe_bool_vector::size() const {
    return data.size();
}

void thd_safe_bool_vector::print() const {
    print_vector<_bool>(data);
}
} } // End namespace kpmeans::base

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

#ifndef __KNOR_THD_SAFE_BOOL_VECTOR_HPP__
#define __KNOR_THD_SAFE_BOOL_VECTOR_HPP__

#include <vector>
#include <memory>

namespace knor { namespace base {

constexpr unsigned LEN = 2;
class _bool {
private:
    char _ [LEN];
public:
    _bool() { }
    _bool(char c);
    operator bool() const;
};


/**
  * \brief This class uses a char[2] vector to represent a
  *     boolean vector since it's represented as a bit vector
  *     which leads to data races for concurrent writers.
*/
class thd_safe_bool_vector {
private:
    std::vector<_bool> data;

    thd_safe_bool_vector(const size_t len) {
        data.resize(len);
    }

    thd_safe_bool_vector(const size_t len, const bool init) :
        thd_safe_bool_vector(len) {
            if (init) {
                for (size_t i = 0; i < data.size(); i++)
                    data[i] = _bool('1');
            } else {
                for (unsigned i = 0; i < data.size(); i++)
                    data[i] = _bool('0');
            }
        }

    void resize(const size_t tosize, const bool init=false);
public:
    typedef std::shared_ptr<thd_safe_bool_vector> ptr;
    static ptr create(const size_t len) {
        return ptr(new thd_safe_bool_vector(len));
    }

    static ptr create(const size_t len, const bool init) {
        return ptr(new thd_safe_bool_vector(len, init));
    }

    const bool get(const size_t idx) const;

    void set(const size_t idx, const bool val);
    bool operator[](const size_t) const;

    size_t size() const;
    void print() const;
    const std::vector<_bool>& raw() const { return data; }
    void check_set(const size_t idx, const bool val);
};
} } // End namespace knor, base
#endif

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

#ifndef __KNOR_HCLUST_COMPACTOR_HPP__
#define __KNOR_HCLUST_COMPACTOR_HPP__

#include <vector>
#include <unordered_map>

namespace knor {
    namespace util {

        class compactor {

            public:
            template <typename T>
            static void remap(std::vector<T>& counts,
                    std::vector<size_t>& remapd_counts,
                    std::unordered_map<unsigned, unsigned>& id_map) {

                for (size_t i = 0; i < counts.size(); i++) {
                    assert(counts[i] >= 0);

                    if (counts[i] > 0) {
                        id_map[i] = remapd_counts.size();
                        remapd_counts.push_back(counts[i]);
                    }
                }
            }
        };
} } // End namespace knor::util
#endif

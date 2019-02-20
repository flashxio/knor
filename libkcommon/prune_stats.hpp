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

#ifndef __KNOR_PRUNE_STATS_HPP__
#define __KNOR_PRUNE_STATS_HPP__

#include <vector>
#include <map>
#include <memory>

namespace knor { namespace base {
// Class to hold stats on the effectiveness of pruning
class prune_stats {
private:
    // Counts per iteration
    size_t lemma1, _3a, _3b, _3c, _4;

    // Total counts
    size_t tot_lemma1, tot_3a, tot_3b, tot_3c, tot_4, iter;
    size_t nrow;
    size_t nclust;

    prune_stats(const size_t nrows, const size_t nclust) {
        _3a = 0; _3b = 0; lemma1 = 0; _3c = 0; _4 = 0;
        tot_lemma1 = 0; tot_3a = 0; tot_3b = 0;
        tot_3c = 0; tot_4 = 0; iter = 0;

        this->nrow = nrows;
        this->nclust = nclust;
    }

public:
    typedef std::shared_ptr<prune_stats> ptr;

    static ptr create(const size_t nrows, const size_t nclust) {
        return ptr(new prune_stats(nrows, nclust));
    }
    void pp_lemma1(const size_t var=1);
    void pp_3a();
    void pp_3b();
    void pp_3c();
    void pp_4();

    const size_t get_lemma1() const;
    const size_t get_3a() const;
    const size_t get_3b() const;
    const size_t get_3c() const;
    const size_t get_4() const;

    prune_stats& operator+=(prune_stats& other);
    void finalize();
    std::vector<double> get_stats();
};

class activation_counter {
    private:
    std::vector<size_t> agg_active_count; // summation of per thread
    std::vector<size_t> active_count;

    activation_counter(const unsigned nthread) {
        active_count.resize(nthread);
    }

    public:
    typedef std::shared_ptr<activation_counter> ptr;
    static ptr create(const unsigned nthread) {
        return ptr(new activation_counter(nthread));
    }

    void active(const unsigned thd);
    void complete();
    std::vector<size_t>& get_active_count_per_iter();
};

class active_counter {
private:
    std::vector<bool> prev_active; // Was a vertex active last iter
    // Active in this iter & active in last
    std::vector<std::vector<bool>> active;
    size_t nrow;

    active_counter(const size_t nrow) {
        this->nrow = nrow;
        prev_active.assign(nrow, false);
        init_iter();
    }

    // Was active in the prev iteration
    const bool was_active(const size_t row) const {
        return prev_active[row];
    }

    void consolidate_samples(std::map<std::vector<bool>, size_t>& data_hash,
            const size_t rows) {
        for (size_t row = 0; row < rows; row++) {
            std::vector<bool> sample(active.size());
            for (size_t iter = 0; iter < active.size(); iter++) {
                sample[iter] = active[iter][row];
            }

            std::map<std::vector<bool>, size_t>::iterator it =
                data_hash.find(sample);
            if (it != data_hash.end())
                it->second++;
            else
                data_hash[sample] = 1;
        }
    }

public:
    typedef std::shared_ptr<active_counter> ptr;
    static ptr create(const size_t nrow) {
        return ptr(new active_counter(nrow));
    }

    // Called at the end of every kmeans iteration
    void init_iter();
    void is_active(const size_t row, const bool val);
    void write_raw(std::string fn, size_t print_row_cnt);
    void write_consolidated(std::string fn, size_t print_row_cnt);
};

} } // End namespace knor::base
#endif


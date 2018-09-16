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

#include <cassert>

#include "prune_stats.hpp"
#include <iostream>

namespace knor { namespace base {

void prune_stats::pp_lemma1(const size_t var) { lemma1 += var; }
void prune_stats::pp_3a() { _3a++; }
void prune_stats::pp_3b() { _3b++; }
void prune_stats::pp_3c() { _3c++; }
void prune_stats::pp_4() { _4++; }

const size_t prune_stats::get_lemma1() const { return lemma1; }
const size_t prune_stats::get_3a() const { return _3a; }
const size_t prune_stats::get_3b() const { return _3b; }
const size_t prune_stats::get_3c() const { return _3c; }
const size_t prune_stats::get_4() const { return _4; }

prune_stats& prune_stats::operator+=(prune_stats& other) {
    lemma1 += other.get_lemma1();
    _3a += other.get_3a();
    _3b += other.get_3b();
    _3c += other.get_3c();
    _4 += other.get_4();
    return *this;
}

void prune_stats::finalize() {
    iter++;
    assert((lemma1 + _3a + _3b + _3c + _4) <=  nrow*nclust);
#ifndef BIND
    std::cout << "\n\nPrune stats count:\n"
        "lemma1 = " << lemma1 << ", 3a = " << _3a
        << ", 3b = " << _3b << ", 3c = " << _3c << ", 4 = " << _4 <<
        std::endl;
#endif

#ifndef BIND
    std::cout << "\n\nPrune stats \%s:\n"
        "lemma1 = " << (lemma1 == 0 ? 0 : ((double)lemma1/(nrow*nclust))*100) <<
        "\%, 3a = " << (_3a == 0 ? 0 : ((double)_3a/(nrow*nclust))*100) <<
        "\%, 3b = " << (_3b == 0 ? 0 : ((double) _3b/(nrow*nclust))*100) <<
        "\%, 3c = " << (_3c == 0 ? 0 : ((double) _3c/(nrow*nclust))*100) <<
        "\%, 4 = " << (_4 == 0 ? 0 : ((double) _4/(nrow*nclust))*100) << "\%"
        << std::endl;
#endif

    tot_lemma1 += lemma1;
    tot_3a += _3a;
    tot_3b += _3b;
    tot_3c += _3c;
    tot_4 += _4;

    lemma1 = 0; _3a = 0; _3b = 0; _3c = 0; _4 = 0; // reset
}

std::vector<double> prune_stats::get_stats() {
    double perc_lemma1 = (tot_lemma1 / ((double)(nrow*iter*nclust)))*100;
    double perc_3a = (tot_3a / ((double)(nrow*iter*nclust)))*100;
    double perc_3b = (tot_3b / ((double)(nrow*iter*nclust)))*100;
    double perc_3c = (tot_3c / ((double)(nrow*iter*nclust)))*100;
    double perc_4 = (tot_4 / ((double)(nrow*iter*nclust)))*100;
    // Total percentage
    double perc = ((tot_3b + tot_3a + tot_3c + tot_4 + tot_lemma1) /
            ((double)(nrow*iter*nclust)))*100;

#ifndef BIND
    printf("tot_lemma1 = %lu, tot_3a = %lu, tot_3b = %lu,"
            " tot_3c = %lu, tot_4 = %lu\n",
            tot_lemma1, tot_3a, tot_3b, tot_3c, tot_4);
#endif

#ifndef BIND
    std::cout << "\n\nPrune stats total:\n"
        "Tot = " << perc << "\%, 3a = " << perc_3a <<
        "\%, 3b = " << perc_3b << "\%, 3c = " << perc_3c
        << "\%, 4 = " << perc_4 << "\%, lemma1 = " << perc_lemma1 << "\%"
        << std::endl;
#endif

    std::vector<double> ret {
        perc_lemma1, perc_3a, perc_3b, perc_3c, perc_4, perc };
    return ret;
}


void activation_counter::active(const unsigned thd) {
    active_count[thd]++;
}

void activation_counter::complete() {
    size_t tot = 0;
    for (std::vector<size_t>::iterator it = active_count.begin();
            it != active_count.end(); ++it)
        tot += *it;

    active_count.assign(active_count.size(), 0); // reset
    agg_active_count.push_back(tot);
}

std::vector<size_t>& activation_counter::get_active_count_per_iter() {
    return agg_active_count;
}

// Called at the end of every kmeans iteration
void active_counter::init_iter() {
    std::vector<bool> v;
    v.assign(nrow, false);
    active.push_back(v); // iteration i all are initially false
}

void active_counter::is_active(const size_t row, const bool val) {
    if (active.size() == 1 && was_active(row)) {
        //knor::base::assert_msg(false, "In first iter the row cannot"
                //" be active previously");
    }

    if (val && was_active(row)) {
        // 1. Grow the rows active vec, to add a true
        active.back()[row] = true;
    } else {
        active.back()[row] = false;
    }

    prev_active[row] = val; // Seen in next iteration
}

void active_counter::write_raw(std::string fn, size_t print_row_cnt) {

    if (print_row_cnt > nrow)
        print_row_cnt = nrow;

    std::string out = "";
    for (size_t row = 0; row < print_row_cnt; row++) {
        for (size_t iter = 0; iter < active.size(); iter++) {
            if (iter == 0)
                out += std::to_string(row) + ", ";

            if (iter + 1 == active.size())
                out += std::to_string(active[iter][row]) + "\n";
            else
                out += std::to_string(active[iter][row]) + ", ";
        }
    }

    FILE* f = fopen(fn.c_str(), "wb");
    fwrite((char*)out.c_str(), out.size(), 1, f);
    fclose(f);

    //printf("Printing the activation:\n%s\n", out.c_str());
}

void active_counter::write_consolidated(std::string fn, size_t print_row_cnt) {
    if (print_row_cnt > nrow)
        print_row_cnt = nrow;

    std::string out = "";
    std::map<std::vector<bool>, size_t> dhash;
    consolidate_samples(dhash, print_row_cnt);

    // Iterate and print
    for (std::map<std::vector<bool>, size_t>::iterator it = dhash.begin();
            it != dhash.end(); ++it) {

        out += std::to_string(it->second) + ", ";
        std::vector<bool> v = it->first;
        for (std::vector<bool>::iterator vit = v.begin();
                vit != v.end(); ++vit) {
            if (vit + 1 == v.end())
                out += std::to_string(*vit) + "\n";
            else
                out += std::to_string(*vit) + ", ";
        }
    }

    FILE* f = fopen(fn.c_str(), "wb");
    fwrite((char*)out.c_str(), out.size(), 1, f);
    fclose(f);

    //printf("Consolidated activation:\n%s\n", out.c_str());
}

} }
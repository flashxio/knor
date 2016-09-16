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

#include <boost/assert.hpp>
#include "convert.hpp"

namespace kpmutil = kpmeans::utils;

int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "usage: ./convert_matrix in_filename to_format"
                " out_filename layout[row/col/rrow/rcol] [nrow] [ncol]\n");
        exit(-1);
    }

    std::string infile = argv[1];
    std::string to_format = argv[2];
    std::string out_filename = argv[3];
    std::string argv4 = std::string(argv[4]);
    kpmutil::conv_layout lay = argv4 == "rrow" ? kpmutil::RAWROW
        : kpmutil::RAWCOL;

    if ((lay == kpmutil::RAWROW || lay == kpmutil::RAWCOL) && argc != 7) {
        fprintf(stderr, "Must provide 2 more args for col-wise\n");
        exit(-1);
    }

    size_t nrow = 0;
    size_t ncol = 0;

    if (argc == 7) {
        nrow = atol(argv[5]);
        ncol = atol(argv[6]);
    }

    std::ofstream out_file;

    if (to_format == "h2o") {
        BOOST_LOG_TRIVIAL(info) << "Converting to " << to_format << " format ...";
        out_file.open(out_filename, std::ios::out);
        if (out_file.is_open()) {
            to_h2o(infile, out_file, lay, nrow, ncol);
            out_file.close();
        } else {
            BOOST_LOG_TRIVIAL(info) << "Failed to open " << out_filename;
            exit(911);
        }
    } else	if (to_format == "spark" || to_format == "dato") {
        BOOST_LOG_TRIVIAL(info) << "Converting to " << to_format << " format ...";
        out_file.open(out_filename, std::ios::out);
        if (out_file.is_open()) {
            to_spark(infile, out_file, lay, nrow, ncol);
            out_file.close();
        } else {
            BOOST_LOG_TRIVIAL(info) << "Failed to open " << out_filename;
            exit(911);
        }
    } else if (to_format == "kmp") {
        BOOST_LOG_TRIVIAL(info) << "Converting to " << to_format << " format ...";
        out_file.open(out_filename, std::ios::out);
        if (out_file.is_open()) {
            fprintf(stderr, "Failed to open file\n");
            to_kmeans_par(infile, out_file, lay, nrow, ncol);
            out_file.close();
        } else {
            BOOST_LOG_TRIVIAL(info) << "Failed to open " << out_filename;
            exit(911);
        }
    } else if (to_format == "fg") {
        BOOST_LOG_TRIVIAL(info) << "Converting to " << to_format << " format ...";
        out_file.open(out_filename, std::ios::binary | std::ios::trunc | std::ios::out);
        if (out_file.is_open()) {
            to_fg(infile, out_file, lay, nrow, ncol);
            BOOST_LOG_TRIVIAL(info) << "Conversion to fg complete";
            out_file.close();
        } else {
            BOOST_LOG_TRIVIAL(info) << "Failed to open " << out_filename;
            exit(911);
        }

    } else {
        fprintf(stderr, "Unknown format '%s'\n", to_format.c_str());
    }

    BOOST_LOG_TRIVIAL(info) << "Conversion complete! file is '" << argv[3] << "'\n";
    return EXIT_SUCCESS;
}

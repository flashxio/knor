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
#include "io.hpp"

namespace kpmbase = kpmeans::base;
namespace kpmeans { namespace utils {

void to_spark(const std::string fn, std::ofstream& of,
        const conv_layout lay, const size_t nrow, const size_t ncol) {

    size_t size = nrow*ncol;
    BOOST_LOG_TRIVIAL(info) << "Malloc-ing matrix with size: " << size;

    double* outmat = new double [size];
    BOOST_LOG_TRIVIAL(info) << "Reading " << fn << ", with r:" << nrow
        << ", c: " << ncol;
    kpmbase::bin_io<double> b(fn, nrow, ncol);
    b.read(outmat);

    BOOST_LOG_TRIVIAL(info) << "Writing matrix ...";
    if (lay == RAWCOL) {
        // Write it
        for (size_t row=0; row < ncol; row++) {
            for (size_t col=0; col < nrow; col++) {
                if (col < nrow-1) {
                    of << outmat[row*nrow+col] << " ";
                } else {
                    of << outmat[row*nrow+col];
                }
            }
            of << "\n";
        }
    } else if (lay == RAWROW) {
        for (size_t row = 0; row < nrow; row++) {
            for (size_t col = 0; col < ncol; col++) {
                if (col < ncol-1) {
                    of << outmat[row*ncol+col] << " ";
                } else {
                    of << outmat[row*ncol+col];
                }
            }
            of << "\n";
        }
    } else { BOOST_VERIFY(false); }
    delete [] outmat;
}

// KMEANS_PAR
void to_kmeans_par(const std::string fn, std::ofstream& of,
        const conv_layout lay, const size_t nrow, const size_t ncol) {
    size_t size = nrow*ncol;
    BOOST_LOG_TRIVIAL(info) << "Malloc-ing matrix with size: " << size;

    double* outmat = new double [size];
    BOOST_LOG_TRIVIAL(info) << "Reading " << fn << ", with r:" << nrow
        << ", c: " << ncol;
    kpmbase::bin_io<double> b(fn, nrow, ncol);
    b.read(outmat);

    BOOST_LOG_TRIVIAL(info) << "Writing matrix ...";

    if (lay == RAWCOL) {
        for (size_t row = 0; row < ncol; row++) {
            of << row+1 << " ";
            for (size_t col = 0; col < nrow; col++) {
                if (col < nrow-1) {
                    of << outmat[row*nrow+col] << " ";
                } else {
                    of << outmat[row*nrow+col];
                }
            }
            of << "\n";
        }
    } else if (lay == RAWROW) {
        for (size_t row = 0; row < nrow; row++) {
            of << row+1 << " ";
            for (size_t col = 0; col < ncol; col++) {
                if (col < ncol-1) {
                    of << outmat[row*ncol+col] << " ";
                } else {
                    of << outmat[row*ncol+col];
                }
            }
            of << "\n";
        }
    } else { BOOST_VERIFY(false); }
    delete [] outmat;
}

void to_fg(const std::string fn, std::ofstream& of,
        const conv_layout lay, const size_t nrow, const size_t ncol) {
    size_t size = nrow*ncol;
    BOOST_LOG_TRIVIAL(info) << "Malloc-ing matrix with size: " << size;

    double* outmat = new double [size];
    BOOST_LOG_TRIVIAL(info) << "Reading " << fn << ", with r:" << nrow
        << ", c: " << ncol;
    kpmbase::bin_io<double> b(fn, nrow, ncol);
    b.read(outmat);

    BOOST_LOG_TRIVIAL(info) << "Writing matrix ...";

    const size_t NUM_ROWS = nrow;
    const size_t NUM_COLS = ncol;
    BOOST_LOG_TRIVIAL(info) << "nrow = " << NUM_ROWS << ", ncol = " << NUM_COLS;

    of.write((char*)&NUM_ROWS, sizeof(size_t)); // size_t rows
    of.write((char*)&NUM_COLS, sizeof(size_t)); // size_t cols
    of.write((char*)&outmat[0], sizeof(double)*size);
    delete [] outmat;
}

void to_h2o(const std::string fn, std::ofstream& of,
        const conv_layout lay, const size_t nrow, const size_t ncol) {

    size_t size = nrow*ncol;
    BOOST_LOG_TRIVIAL(info) << "Malloc-ing matrix with size: " << size;

    double* outmat = new double [size];
    BOOST_LOG_TRIVIAL(info) << "Reading " << fn << ", with r:" << nrow
        << ", c: " << ncol;
    kpmbase::bin_io<double> b(fn, nrow, ncol);
    b.read(outmat);

    BOOST_LOG_TRIVIAL(info) << "Writing matrix ...";
    if (lay == RAWCOL) {
        // Write it
        for (size_t row = 0; row < ncol; row++) {
            of << row + 1;
            for (size_t col = 0; col < nrow; col++) {
                of << "," << outmat[row*nrow+col];
            }
            of << "\n";
        }
    } else if (lay == RAWROW) {
        for (size_t row = 0; row < nrow; row++) {
            of << row + 1;
            for (size_t col = 0; col < ncol; col++) {
                of << "," << outmat[row*ncol+col];
            }
            of << "\n";
        }
    } else { BOOST_VERIFY(false); }
    delete [] outmat;
}

} }

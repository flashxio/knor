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

#ifndef __KNOR_LINALG_HPP__
#define __KNOR_LINALG_HPP__

#include<bits/stdc++.h>

// Adapted from: https://www.geeksforgeeks.org/adjoint-inverse-matrix/

namespace knor { namespace base {

    class linalg {
        public:
        static void getCofactor(double* A, double* temp,
                double p, size_t q, double n, const size_t N) {
            size_t i = 0, j = 0;

            // Looping for each element of the matrix
            for (size_t row = 0; row < n; row++) {
                for (size_t col = 0; col < n; col++) {
                    // Copying into temporary matrix only those element
                    // which are not in given row and column
                    if (row != p && col != q) {
                        temp[(i*N)+j++] = A[row*N+col];

                        // Row is filled, so increase row index and
                        // reset col index
                        if (j == n - 1) {
                            j = 0;
                            i++;
                        }
                    }
                }
            }
        }

        /* Recursive function for finding determinant of matrix.
           n is current dimension of A[][]. */
        static double determinant(double* A, size_t n, const size_t N) {
            double D = 0; // Initialize result

            // Base case : if matrix contains single element
            if (n == 1)
                return A[0];

            double temp[N*N]; // To store cofactors

            double sign = 1; // To store sign multiplier

            // Iterate for each element of first row
            for (size_t f = 0; f < n; f++) {
                // Getting Cofactor of A[0][f]
                getCofactor(A, temp, 0, f, n, N);
                D += sign * A[f] * determinant(temp, n - 1, N);

                // terms are to be added with alternate sign
                sign = -sign;
            }

            return D;
        }

        // Function to get adjoint of A[N][N] in adj[N][N].
        static void adjoint(double* A, double* adj, const size_t N) {
            if (N == 1) {
                adj[0] = 1;
                return;
            }

            // temp is used to store cofactors of A[][]
            double sign = 1, temp[N*N];

            for (size_t i=0; i<N; i++) {
                for (size_t j=0; j<N; j++) {
                    // Get cofactor of A[i][j]
                    getCofactor(A, &temp[0], i, j, N, N);

                    // sign of adj[j][i] positive if sum of row
                    // and column indexes is even.
                    // Interchanging rows and columns to get the
                    // transpose of the cofactor matrix
                    adj[j*N+i] = (sign)*(determinant(temp, N-1, N));
                }
            }
        }

        // Function to calculate and store inverse, returns false if
        // matrix is singular
        static bool inverse(double* A, double* inverse, const size_t N) {
            // Find determinant of A[][]
            double det = determinant(A, N, N);
            if (det == 0) {
                printf("Singular matrix, can't find its inverse\n");
                return false;
            }

            // Find adjoint
            double adj[N*N];
            adjoint(A, adj, N);

            // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
            for (size_t i=0; i<N; i++)
                for (size_t j=0; j<N; j++)
                    inverse[i*N+j] = adj[i*N+j]/det;

            return true;
        }
    };
} }
#endif

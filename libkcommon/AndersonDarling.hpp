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

#ifndef __KNOR_ANDERSON_DARLING_HPP__
#define __KNOR_ANDERSON_DARLING_HPP__

#include <cmath>
#include <algorithm>
#include <numeric>

namespace knor { namespace base {

class AndersonDarling {
private:
	static const double phi(const double x) {
		return 0.5 * erfc(-x * M_SQRT1_2);
	}

public:
	static void compute_critical_values(double N, std::vector<double>& out) {
		std::vector<double> _Avals_norm { 0.576, 0.656, 0.787, 0.918, 1.092 };
		const double divisor = 1.0 + 4.0/N - 25.0/(N*N);
		for (auto const& i : _Avals_norm) {
			out.push_back(i / divisor);
		}
	}

	static double compute_statistic(const size_t n, double* X) {
        std::sort(X, X + n);
        double X_avg = std::accumulate(X, X + n, 0.0)
            / static_cast<double>(n);

		// Find the variance of X
		double X_sig = 0.0;
		for (size_t i = 0; i < n; i++) {
            auto diff = (X[i] - X_avg);
			X_sig += diff * diff;
		}

		X_sig /= (n-1);
        X_sig = std::sqrt(X_sig);

		// The values X_i are standardized to create new values Y_i
        std::vector<double> Y(n);
		for (size_t i = 0; i < n; i++) {
			Y[i] = (X[i] - X_avg)/(X_sig);
		}

		// With a standard normal CDF, we calculate the Anderson_Darling Statistic
		double A = -((long)n);
		for (size_t i = 0; i < n; i++) {
			A +=  -1.0 / static_cast<double>(n) *(2*(i + 1) - 1) *
				(std::log(phi(Y[i])) + std::log(1 - phi(Y[n - 1 - i])));
		}

		return A;
	}

	static void compute_statistic(std::vector<double>& X) {
		compute_statistic(X.size(), &X[0]);
	}
};

}} // End namespace knor::base
#endif

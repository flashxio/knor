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

#include <vector>
#include <cassert>
#include <stdio.h>

#include "io.hpp"
#include "AndersonDarling.hpp"

using namespace knor::core;

int main() {
	std::vector<double> x {1, 2, 3, 4};

	const double EPS = 0.00001;

	auto ad = AndersonDarling::compute_statistic(4, &x[0]);
	const double AD_RES = 0.159201;
	printf("AD: %.5f\n", ad);
	assert(ad - AD_RES < EPS);

	std::vector<double> cv;
	std::vector<double> CV_RES {1.31657, 1.49943, 1.79886, 2.09829, 2.496};
	AndersonDarling::compute_critical_values(x.size(), cv);

	printf("Critical values: \n");
	print(cv);

	assert(cv.size() == CV_RES.size());
	for (size_t i = 0; i < cv.size(); i++)
		assert(cv[i] - CV_RES[i] < EPS);

	printf("\nAndersonDarling test successful!\n");
}

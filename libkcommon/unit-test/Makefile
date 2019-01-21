# Copyright 2016 neurodata (http://neurodata.io/)
# Written by Disa Mhembere (disa@jhu.edu)
#
# This file is part of knor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include ../../Makefile.common

LDFLAGS := -L.. -lkcommon $(LDFLAGS)
CXXFLAGS := -I.. $(CXXFLAGS)

TESTFILES := test_thd_safe_bool_vector test_clusters test_reader\
	test_dist_matrix test_dense_matrix test_linalg testeigen test_util\
	test_types

all: $(TESTFILES)

test: all
	./test_clusters
	./test_thd_safe_bool_vector 2 500
	./test_reader
	./test_dist_matrix
	./test_dense_matrix
	./test_util
	./test_types

test_thd_safe_bool_vector: test_thd_safe_bool_vector.o ../libkcommon.a
	$(CXX) -o test_thd_safe_bool_vector test_thd_safe_bool_vector.o $(LDFLAGS)

test_types: test_types.o ../libkcommon.a
	$(CXX) -o test_types test_types.o $(LDFLAGS)

test_clusters: test_clusters.o ../libkcommon.a
	$(CXX) -o test_clusters test_clusters.o $(LDFLAGS)

test_dist_matrix: test_dist_matrix.o ../libkcommon.a
	$(CXX) -o test_dist_matrix test_dist_matrix.o $(LDFLAGS)

test_reader: test_reader.o ../libkcommon.a
	$(CXX) -o test_reader test_reader.o $(LDFLAGS)

test_dense_matrix: test_dense_matrix.o ../libkcommon.a
	$(CXX) -o test_dense_matrix test_dense_matrix.o $(LDFLAGS)

test_linalg: test_linalg.o ../libkcommon.a
	$(CXX) -o test_linalg test_linalg.o $(LDFLAGS)

testeigen: testeigen.o ../libkcommon.a
	$(CXX) -o testeigen testeigen.o $(LDFLAGS)

test_util: test_util.o ../libkcommon.a
	$(CXX) -o test_util test_util.o $(LDFLAGS)

clean:
	rm -f *.d
	rm -f *.o
	rm -f *~
	rm -f $(TESTFILES)

-include $(DEPS)

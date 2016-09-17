# Copyright 2016 neurodata (http://neurodata.io/)
# Written by Disa Mhembere (disa@jhu.edu)
#
# This file is part of k-par-means.
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

include ../Makefile.common

all: create_matrix convert_matrix test

LDFLAGS += -L../libcommon -lcommon
CXXFLAGS += -I../libcommon

convert_matrix: convert.o convert_matrix.o
	$(CXX) -o convert_matrix convert.o convert_matrix.o $(LDFLAGS)

create_matrix: create_matrix.o
	$(CXX) -o create_matrix create_matrix.o $(LDFLAGS)

test: convert_matrix
	$(MAKE) -C test

clean:
	rm -f *.d
	rm -f *.o
	rm -f *~
	rm -f convert_matrix
	rm -f create_matrix
	make --ignore-errors -C test clean

-include $(DEPS)

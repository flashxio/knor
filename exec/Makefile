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

LDFLAGS := -L../libauto -lauto -L../libman -lman \
	-L../libcommon -lcommon $(LDFLAGS)
CXXFLAGS += -I.. -I../libauto -I../libman -I../libdist -I../libcommon
DIST_FLAGS := -L../libdist -ldist

all: kpmeans dist_kmeans

kpmeans: kpmeans.o
	$(CXX) -o kpmeans kpmeans.o $(LDFLAGS)

dist_kmeans: dist_kmeans.o
	$(CXX) -o dist_kmeans dist_kmeans.o $(LDFLAGS) $(DIST_FLAGS)

clean:
	rm -f *.d
	rm -f *.o
	rm -f *~
	rm -f kpmeans
	rm -f dist_kmeans

-include $(DEPS)

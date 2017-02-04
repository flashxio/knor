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

include ../../Makefile.common

LDFLAGS := -L.. -lman -L../../libkcommon -lkcommon $(LDFLAGS)
CXXFLAGS := -I.. -I../../libkcommon $(CXXFLAGS)

TESTFILES := test_kmeans_thread test_task_queue test_kmeans_task_thread

all: $(TESTFILES)

test: all
	./test_kmeans_task_thread 2
	./test_kmeans_thread 2
	./test_task_queue 2

test_kmeans_task_thread: test_kmeans_task_thread.o
	$(CXX) -o test_kmeans_task_thread test_kmeans_task_thread.o $(LDFLAGS)

test_kmeans_thread: test_kmeans_thread.o
	$(CXX) -o test_kmeans_thread test_kmeans_thread.o $(LDFLAGS)

test_task_queue: test_task_queue.o
	$(CXX) -o test_task_queue test_task_queue.o $(LDFLAGS)

clean:
	rm -f *.d
	rm -f *.o
	rm -f *~
	rm -f $(TESTFILES)

-include $(DEPS)

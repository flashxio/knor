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

UNAME_S := $(shell uname -s)

all:
ifeq ($(UNAME_S), Linux)
	$(MAKE) im dist
else
	$(MAKE) im
endif

im:
	$(MAKE) -f Makefile.im

dist:
	$(MAKE) -f Makefile.dist

clean:
	rm -f *.d
	rm -f *.o
	rm -f *~
	rm -f knori
	rm -f knord

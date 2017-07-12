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

ifeq ($(UNAME_S), Darwin)
LDFLAGS := -L../../libman -lman -L../../libkcommon -lkcommon $(LDFLAGS)
CXXFLAGS += -I.. -I../../libman -I../../libkcommon
else
LDFLAGS := -L.. -L../../libauto -lauto -L../../libman -lman \
	-L../../libkcommon -lkcommon $(LDFLAGS)
CXXFLAGS += -I.. -I../../libauto -I../../libman -I../../libkcommon
endif

TESTFILES := test_knori

all: $(TESTFILES)

test: all
	./test_knori

test_knori: test_knori.o
	$(CXX) -o test_knori test_knori.o $(LDFLAGS)

clean:
	rm -f *.d
	rm -f *.o
	rm -f *~
	rm -f $(TESTFILES)

-include $(DEPS)

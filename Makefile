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

include Makefile.common

all: build_lib utils

build_lib:
	$(MAKE) -C libcommon
	$(MAKE) -C libauto
	$(MAKE) -C libman
	#$(MAKE) -C process

utils: build_lib
	$(MAKE) -C utils

clean:
	rm -f *.d
	rm -f *.o
	rm -f *~
	make --ignore-errors -C libcommon clean
	make --ignore-errors -C utils clean
	make --ignore-errors -C libauto clean
	make --ignore-errors -C libman clean
	#make --ignore-errors -C process clean

-include $(DEPS)

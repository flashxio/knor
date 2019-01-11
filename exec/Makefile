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

UNAME_S := $(shell uname -s)
OUTDIR_IM := outdir-IM
OUTDIR_DM := outdir-DM
OUTDIR_FDM := outdir-FDM

TESTFILES := knori knors knord skmeans medoids fcm hclust mb_knori

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

test-dist:
	@echo "Running IM for checking .."
	./knori ../test-data/matrix_r50_c5_rrw.bin 50 5 8 \
		-t none -C ../test-data/init_clusters_k8_c5.bin -T 2 -o $(OUTDIR_IM)
	@echo "Running DIST .."
	mpirun.mpich -n 2 ./knord ../test-data/matrix_r50_c5_rrw.bin 50 5 8 \
		-t none -C ../test-data/init_clusters_k8_c5.bin -T 2 -o $(OUTDIR_DM)
	mpirun.mpich -n 2 ./knord ../test-data/matrix_r50_c5_rrw.bin 50 5 8 \
		    -t none -C ../test-data/init_clusters_k8_c5.bin -T 2 \
			-P -o $(OUTDIR_FDM)
	./test-outdiff.sh
	@echo "Cleaning up ..."
	rm -rf $(OUTDIR_IM) $(OUTDIR_DM) $(OUTDIR_FDM)

test-nor:
	@echo "Testing medoids!"
	./medoids -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 32 -i 10
	@echo "Testing skmeans"
	./skmeans -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 32 -i 10
	@echo "Testing fcm"
	./fcm -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 32 -i 10
	@echo "Testing hclust"
	./hclust -f ../test-data/iris.bin -n 150 -m 4 -k 4 -T 32 -i 4 -l .2
	#@echo "Testing mb_knori"
	#./mb_knori -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 2 -i 10 -M 30

test:
	$(MAKE) test-nor
	$(MAKE) test-dist

clean:
	rm -f *.d
	rm -f *.o
	rm -f *~
	rm -f $(TESTFILES)

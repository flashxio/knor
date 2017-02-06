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

OUTDIR_IM=outdir-IM
OUTDIR_SEM=outdir-SEM
OUTDIR_FSEM=outdir-FSEM

cd .. &&
    echo "root_conf=$(pwd)/libsem/FlashX/flash-graph/conf/data_files.txt" >> \
    libsem/FlashX/flash-graph/conf/run_graph.txt &&
    echo "0:$(pwd)/release-test" >> \
    libsem/FlashX/flash-graph/conf/data_files.txt

    exec/knori test-data/matrix_r50_c5_rrw.bin 50 5 8 \
    -t none -C test-data/init_clusters_k8_c5.bin -T 2 -o $OUTDIR_IM &&
    exec/knors libsem/FlashX/flash-graph/conf/run_graph.txt\
    test-data/matrix_r50_c5_rrw.adj 50 5 8 \
    -t none -C test-data/init_clusters_k8_c5.bin -o $OUTDIR_SEM &&
    if [ "$(diff $OUTDIR_IM/* $OUTDIR_SEM/*)" = "" ];
then
    echo "knors PRUNED test success!"
else
    echo "knors PRUNED release test failure!"
    exit 1
fi

exec/knors libsem/FlashX/flash-graph/conf/run_graph.txt\
    test-data/matrix_r50_c5_rrw.adj 50 5 8 \
    -t none -C test-data/init_clusters_k8_c5.bin -P -o $OUTDIR_SEM &&
    if [ "$(diff $OUTDIR_IM/* $OUTDIR_FSEM/*)" = "" ];
    then
        echo "knors FULL test success!"
    else
        echo "knors FULL release test failure!"
        exit 1
    fi

# Cleanup
echo "Cleaning up ..."
rm -rf $OUTDIR_IM $OUTDIR_SEM $OUTDIR_FSEM

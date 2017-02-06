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
OUTDIR_DM=outdir-DM
OUTDIR_FDM=outdir-FDM

cd .. &&
    exec/knori test-data/matrix_r50_c5_rrw.bin 50 5 8 \
     -t none -C test-data/init_clusters_k8_c5.bin -T 2 -o $OUTDIR_IM &&
 mpirun.mpich -n 2 exec/knord test-data/matrix_r50_c5_rrw.bin 50 5 8 \
    -t none -C test-data/init_clusters_k8_c5.bin -T 2 -o $OUTDIR_DM &&
    if [ "$(diff $OUTDIR_IM/* $OUTDIR_DM/*)" = "" ];
then
    echo "knord PRUNED test success!"
else
    echo "knord PRUNED release test failure!"
    exit 1
fi

mpirun.mpich -n 2 exec/knord test-data/matrix_r50_c5_rrw.bin 50 5 8 \
    -t none -C test-data/init_clusters_k8_c5.bin -T 2 -P -o $OUTDIR_FDM &&
    if [ "$(diff $OUTDIR_IM/* $OUTDIR_FDM/*)" = "" ];
    then
        echo "knord FULL test success!"
    else
        echo "knord FULL release test failure!"
        exit 1
    fi

# Cleanup
echo "Cleaning up ..."
rm -rf $OUTDIR_IM $OUTDIR_DM $OUTDIR_FDM

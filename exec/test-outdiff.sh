#!/usr/bin/env bash
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

OUTDIR_IM=outdir-IM
OUTDIR_DM=outdir-DM
OUTDIR_FDM=outdir-FDM

if [ "$(diff $OUTDIR_IM/* $OUTDIR_DM/*)" = "" ];
then
    echo "knord PRUNED test success!"
else
    echo "knord PRUNED release test failure!"
    exit 1
fi

if [ "$(diff $OUTDIR_IM/* $OUTDIR_FDM/*)" = "" ];
then
    echo "knord FULL test success!"
else
    echo "knord FULL release test failure!"
    exit 1
fi

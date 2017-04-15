#!/usr/bin/env bash
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

# Ubuntu install script

cd $HOME
# Make sure the package information is up-to-date
apt-get update
apt-get sudo apt-get -o Dpkg::Options::="--force-confold" --force-yes -y upgrade

apt-get install -y build-essential
# In memory dependencies
apt-get install -y libboost-all-dev
# NUMA
apt-get install -y libnuma-dbg libnuma-dev libnuma1

# Message Passing Interface
apt-get install -y libmpich-dev

# SEM
apt-get install -y libaio-dev
apt-get install -y libatlas-base-dev
apt-get install -y libgoogle-perftools-dev

#if [ $(dpkg-query -W -f='${Status}' libmpich2-dev 2>/dev/null | grep -c "ok installed")
#    -eq 0 ]; then
#    apt-get install libmpich2-dev;
#fi

# Source control
apt-get install -y git

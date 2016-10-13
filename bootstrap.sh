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

NPROC=`nproc`
cd $HOME
# In memory dependencies
apt-get install libboost-all-dev

# NUMA
apt-get install -y libnuma-dbg libnuma-dev libnuma1

# Boost
apt-get install -y libboost-all-dev

# Make sure the package information is up-to-date
apt-get update
apt-get -y upgrade

## Qt5 visualization
apt-get install -y qt5-default
#apt-get install -y imagemagick

# Python visualization
apt-get install -y python-matplotlib python-networkx

# Compilers
apt-get install -y gfortran-4.8 clang-3.5

# Message Passing Interface
apt-get install -y libmpich2-dev

# Configuration
apt-get install -y cmake

# Source control
apt-get install -y git

# Check out Elemental
git clone https://github.com/elemental/Elemental
mkdir build
cd build
cmake ../Elemental
make -j $NPROC
make install

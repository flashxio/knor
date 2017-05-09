#!/usr/bin/env python

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

from distutils.core import setup, Extension

knori_module = Extension('_kmeans_task_coordinator',
                           sources=['kmeans_task_coordinator_wrap.cpp',
                               'kmeans_task_coordinator.cpp'],
                           extra_compile_args=["-std=gnu++11", "-O3",
                               "-I.",
                               "-I../libkcommon", "-I..", "-fPIC",
                               "-DSTATISTICS", "-DBOOST_LOG_DYN_LINK",
                               "-fopenmp"],
                           extra_link_args=["-L../libman", "-lman", "-L../libkcommon",
                               "-lkcommon", "-lnuma", "-lpthread", "-fopenmp",
                               "-lboost_log", "-rdynamic", "-lrt", "-rdynamic",
                               "-lhwloc", "-lpython2.7"]
                           )

setup (name = '_kmeans_task_coordinator',
       version = '0.1',
       author      = "Disa Mhembere",
       maintainer  = "Disa Mhembere",
       description = """`knori` Python wrapper""",
       ext_modules = [knori_module],
       py_modules = ["kmeans_task_coordinator"],
       )

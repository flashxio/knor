#!/usr/bin/env python

import os
import sys, re
from glob import glob
from exceptions import NotImplementedError
from exceptions import RuntimeError
from distutils.errors import DistutilsSetupError
from distutils.command.build_clib import build_clib
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

_REPO_ISSUES_ = "https://github.com/flashxio/knorPy/issues"
_OS_SUPPORTED_ = {"linux":"linux", "mac":"darwin"}


patts = []
for opsys in _OS_SUPPORTED_.itervalues():
    patts.append(re.compile("(.*)("+opsys+")(.*)"))

raw_os = sys.platform.lower()

OS = None
for patt in patts:
    res = re.match(patt, raw_os)
    if res is not None:
        OS = res.groups()[1]
        break

if OS not in _OS_SUPPORTED_.values():
    raise RuntimeError("Operating system {}\n." +\
            "Please post an issue at {}\n".format(raw_os, _REPO_ISSUES_))


# Hack to stop -Wstrict-prototypes warning on linux
if OS == _OS_SUPPORTED_["linux"]:
    from distutils.sysconfig import get_config_vars

    (opt,) = get_config_vars('OPT')
    os.environ['OPT'] = " ".join(
                flag for flag in opt.split() if flag != '-Wstrict-prototypes'
                )

################################ End VarDecl ###################################

# For C++ libraries
libkcommon = ("kcommon",
        {"sources": glob(os.path.join("libkcommon/", "*.cpp"))})
libman = ('man',
        {'sources': glob(os.path.join("libman/", "*.cpp"))})

# Minimum libraries we will build
libraries = [libkcommon, libman]

if OS == _OS_SUPPORTED_["linux"]:
    libauto = ("auto",
            {"sources": glob(os.path.join("libauto/", "*.cpp"))})
    libraries.append(libauto)

# Add some sources + cython modules
sources = glob(os.path.join("binding/", "*.cpp"))
sources.append("python/knor/knor.pyx")

# Compile & link arguments
if OS == _OS_SUPPORTED_["linux"]:
    extra_compile_args = ["-std=c++11", "-O3", "-fPIC",
            "-Wno-attributes", "-Wno-unused-variable",
            "-Wno-unused-function", "-I.","-Ilibman", "-Ilibauto",
            "-Ibinding", "-Ilibkcommon",
            "-DBOOST_LOG_DYN_LINK", "-fopenmp",
            "-I/usr/local/lib/python2.7/dist-packages/numpy/core/include",
            "-I/usr/include/python2.7",
            "-DBIND", "-DLINUX"]

    extra_link_args=["-Llibman", "-lman", "-Llibauto", "-lauto",
            "-Llibkcommon", "-lkcommon", "-lpthread", "-lnuma", "-fopenmp",
            "-lboost_log", "-lboost_system"
            ]

elif OS == _OS_SUPPORTED_["mac"]:
    extra_compile_args = ["-std=c++11", "-O3",
            "-Wno-unused-function", "-I.","-Ilibman",
            "-Ibinding", "-Ilibkcommon",
            "-I/usr/local/lib/python2.7/site-packages/numpy/core/include",
            "-DBOOST_LOG_DYN_LINK", "-I/usr/local/opt/boost/include",
            "-DBIND", "-DOSX"]

    extra_link_args=["-Llibman", "-lman", "-Llibkcommon",
            "-lkcommon", "-lpthread", "-lboost_log-mt",
            "-lboost_system", "-L/usr/local/opt/boost/lib",
            ]
else:
    raise RuntimeError("Unsupported OS {}".format(raw_os))

# Build cython modules
ext_modules = cythonize(Extension(
        "knor.knor",                                # the extension name
        sources=sources,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args))

################################################################################

class knor_clib(build_clib, object):
    def initialize_options(self):
        super(knor_clib, self).initialize_options()
        if OS == _OS_SUPPORTED_["linux"]:
            self.include_dirs = [
                "libman", "libauto", "binding", "libkcommon",
                ]
            self.define = [
                    ("BOOST_LOG_DYN_LINK", None),
                    ("BIND", None), ("LINUX", None)
                    ]
        elif OS == _OS_SUPPORTED_["mac"]:
            self.include_dirs = [
                "libman", "binding", "libkcommon",
                "/usr/local/opt/boost/include",
                "/usr/local/lib/python2.7/site-packages/numpy/core/include",
                ]

            self.define = [
                    ("BOOST_LOG_DYN_LINK", None),
                    ("BIND", None), ("OSX", None)
                    ]

        else:
            raise RuntimeError("Unsupported OS {}".format(raw_os))

    def build_libraries(self, libraries):
        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError, \
                      ("in 'libraries' option (library '%s'), " +
                       "'sources' must be present and must be " +
                       "a list of source filenames") % lib_name
            sources = list(sources)

            print "building '%s' library" % lib_name

            # First, compile the source code to object files in the library
            # directory.  (This should probably change to putting object
            # files in a temporary build directory.)
            macros = build_info.get('macros')
            include_dirs = build_info.get('include_dirs')

            # pass flasgs to compiler
            extra_preargs = ["-std=c++11", "-O3", "-Wno-unused-function"]

            if OS == _OS_SUPPORTED_["linux"]:
                extra_preargs.append("-fopenmp")

            objects = self.compiler.compile(sources,
                                            output_dir=self.build_temp,
                                            macros=macros,
                                            include_dirs=include_dirs,
                                            debug=self.debug,
                                            extra_preargs=extra_preargs)

            # Now "link" the object files together into a static library.
            # (On Unix at least, this isn't really linking -- it just
            # builds an archive.  Whatever.)
            self.compiler.create_static_lib(objects, lib_name,
                                            output_dir=self.build_clib,
                                            debug=self.debug)

################################################################################

# Run the setup
setup(
    name="knor",
    version="0.0.1a10",
    description="A fast parallel k-means library for Linux and Mac",
    long_description="The k-means NUMA Optimized Routine library or " +\
    "knor is a highly optimized and fast library for computing " +\
    "k-means in parallel with accelerations for Non-Uniform Memory " +\
    "Access (NUMA) architectures",
    url="https://github.com/flashxio/knor",
    author="Disa Mhembere",
    author_email="disa@jhu.edu",
    license="Apache License, Version 2.0",
    keywords="kmeans k-means parallel clustering machine-learning",
    install_requires=[
        "numpy",
        "Cython==0.23.5",
        "cython==0.23.5",
        ],
    package_dir = {"knor": os.path.join("python", "knor")},
    packages=["knor", "knor.Exceptions"],
    libraries =libraries,
    cmdclass = {'build_clib': knor_clib, 'build_ext': build_ext},
    ext_modules = ext_modules,
    )

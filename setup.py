#!/usr/bin/env/python

import numpy
import setuptools
from setuptools.command.build_py import build_py as _build_py
import os
import hashlib
from setuptools.command.build_ext import build_ext as _build_ext


conda_prefix = os.environ.get("CONDA_PREFIX", "/usr")  # fallback if not in conda


NLOP_LIB_DIRS = [
    '/opt/homebrew/lib',                    # MacOS BS
    #'/opt/homebrew/Cellar/nlopt/2.7.1/lib', # For ARM
    #'/usr/local/Cellar/nlopt/2.7.1/lib',    # For x86
    '/usr/lib',                             # system library path
    '/usr/local/lib',                       # custom lib path
    os.path.join(conda_prefix, 'lib'),
]

NLOP_INCLUDE_DIRS = [
    '/opt/homebrew/include',
    #'/opt/homebrew/Cellar/nlopt/2.7.1/include',    # For ARM
    #'/usr/local/Cellar/nlopt/2.7.1/include',       # For x86
    '/usr/include',                                # system includes (e.g. nlopt.hpp)
    '/usr/local/include',                          # custom install path
    'include',                                     # local project includes
    os.path.join(conda_prefix, 'include'),
]

INCLUDE_DIRS = NLOP_INCLUDE_DIRS + [numpy.get_include()]

spires = setuptools.Extension(
    name='spires_inversion._core',
    sources=['spires_inversion/spires.i', 'spires_inversion/spires.cpp'],
    swig_opts=['-c++'],
    extra_compile_args=['-std=c++11'],
    library_dirs=NLOP_LIB_DIRS,    
    include_dirs=INCLUDE_DIRS,     
    libraries=['nlopt'],
    language='c++'
)


def source_digest(root="spires_inversion"):
    h = hashlib.sha256()
    for name in ("spires.cpp", "spires.h"):
        with open(os.path.join(root, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


class build_ext(_build_ext):
    def run(self):
        super().run()
        with open(os.path.join("spires_inversion", "_core.src.sha256"), "w") as f:
            f.write(source_digest() + "\n")


class build_py(_build_py):

    def run(self):
        """ 
        We need to overwrite run to make sure extension is built before getting copied
        """
        self.run_command("build_ext")
        return super().run()


setuptools.setup(
    packages=setuptools.find_packages(),
    ext_modules=[spires],
    cmdclass={'build_py': build_py, 'build_ext': build_ext},
)
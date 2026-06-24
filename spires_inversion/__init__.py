from spires_inversion.invert import *
from spires_inversion.interpolator import *
from spires_inversion.parallel import speedy_invert_dask, encode_results
import spires_inversion.legacy
import spires_inversion.parallel

# Version from setuptools_scm
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spires_inversion")
except PackageNotFoundError:
    __version__ = "unknown"

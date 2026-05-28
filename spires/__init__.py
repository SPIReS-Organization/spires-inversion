from spires.invert import *
from spires.interpolator import *
from spires.parallel import speedy_invert_dask, encode_results
import spires.legacy
import spires.parallel

# Version from setuptools_scm
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spires")
except PackageNotFoundError:
    __version__ = "unknown"

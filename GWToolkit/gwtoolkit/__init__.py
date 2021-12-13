"""
GWToolkit: A Python Toolbox for gravitational wave astronomy
================================================
Documentation is available in the docstrings and
online at https://....
Contents
--------
SciPy imports all the functions from the NumPy namespace, and in
addition provides:
Subpackages
-----------
Using any of these subpackages requires an explicit import. For example,
``import GWToolkit.cluster``.


::
 cluster                      --- Vector Quantization / Kmeans
 fft                          --- Discrete Fourier transforms
 fftpack                      --- Legacy discrete Fourier transforms
 integrate                    --- Integration routines
 ...

Utility tools
-------------
::
 test              --- Run scipy unittests
 show_config       --- Show scipy build configuration
 show_numpy_config --- Show numpy build configuration
 __version__       --- GWToolkit version string
 __scipy_version__ --- SciPy version string
 __numpy_version__ --- Numpy version string

REF: https://github.com/scipy/scipy/blob/master/scipy/__init__.py
"""
from .gw import waveform
from .torch import dataset
from .utils import *
import os
import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import c_float, c_int, c_uint16, c_uint32
import importlib_resources
from pathlib import Path

def aligned_zeros(shape, boundary=16, dtype=float, order='C'):
      N = np.prod(shape)
      d = np.dtype(dtype)
      tmp = np.zeros(N * d.itemsize + boundary, dtype=np.uint8)
      address = tmp.__array_interface__['data'][0]
      offset = (boundary - address % boundary) % boundary
      return tmp[offset:offset+N*d.itemsize]\
                 .view(dtype=d)\
                 .reshape(shape, order=order)

c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
c_uint16_p = ctypes.POINTER(ctypes.c_uint16)
c_uint32_p = ctypes.POINTER(ctypes.c_uint32)
c_uint64_p = ctypes.POINTER(ctypes.c_uint64)

# load the library, using numpy mechanisms
with importlib_resources.path("minerva_lib", "crender") as libpath:
    path = Path(str(libpath)).parent.parent
    crender = npct.load_library("crender", str(path))

# setup the return types and argument types
crender.rescale_intensity16.restype = None
crender.rescale_intensity16.argtypes = [c_uint16_p, c_uint16, c_uint16, c_int]
crender.rescale_intensity32.restype = None
crender.rescale_intensity32.argtypes = [c_uint32_p, c_uint32, c_uint32, c_int]

crender.clip16.restype = None
crender.clip16.argtypes = [c_uint16_p, c_uint16, c_uint16, c_int]
crender.clip32.restype = None
crender.clip32.argtypes = [c_uint32_p, c_uint32, c_uint32, c_int]
crender.clip32_conv8.restype = None
crender.clip32_conv8.argtypes = [c_uint32_p, c_uint8_p, c_int]
crender.clip32_conv8_32.restype = None
crender.clip32_conv8_32.argtypes = [c_uint64_p, c_uint8_p, c_int]

crender.composite16.restype = None
crender.composite16.argtypes = [c_uint32_p, c_uint16_p, c_float, c_float, c_float, c_int]
crender.composite32.restype = None
crender.composite32.argtypes = [c_uint64_p, c_uint32_p, c_float, c_float, c_float, c_int]

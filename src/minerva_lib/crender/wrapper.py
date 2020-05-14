import sys
import logging
import numpy as np
import os

import numpy.ctypeslib as npct
import ctypes
from ctypes import c_float, c_int, c_uint16

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

# load the library, using numpy mechanisms
#crender = npct.load_library("crender", os.path.dirname(__file__))
crender = npct.load_library("crender", ".")

# setup the return types and argument types
crender.rescale_intensity.restype = None
crender.rescale_intensity.argtypes = [c_float_p, c_float, c_float, c_int]
crender.rescale_intensity16.restype = None
crender.rescale_intensity16.argtypes = [c_uint16_p, c_int, c_int, c_int]

crender.clip.restype = None
crender.clip.argtypes = [c_float_p, c_float, c_float, c_int, c_int]
crender.clip16.restype = None
crender.clip16.argtypes = [c_uint16_p, c_uint16, c_uint16, c_int, c_int]
crender.clip32_conv8.restype = None
crender.clip32_conv8.argtypes = [c_uint32_p, c_uint8_p, c_uint16, c_uint16, c_int, c_int]

crender.image_as_float.restype = c_float_p
crender.image_as_float.argtypes = [c_uint16_p, c_int]

crender.composite.restype = None
crender.composite.argtypes = [c_float_p, c_float_p, c_float, c_float, c_float, c_int]
crender.composite16.restype = None
crender.composite16.argtypes = [c_uint32_p, c_uint16_p, c_float, c_float, c_float, c_int]
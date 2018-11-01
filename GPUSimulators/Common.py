# -*- coding: utf-8 -*-

"""
This python module implements the different helper functions and 
classes

Copyright (C) 2018  SINTEF ICT

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os

import numpy as np
import time
import re
import io
import hashlib
import logging
import gc

import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda


"""
Class which keeps track of time spent for a section of code
"""
class Timer(object):
    def __init__(self, tag, log_level=logging.DEBUG):
        self.tag = tag
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000 # millisecs
        self.logger.log(self.log_level, "%s: %f ms", self.tag, self.msecs)

    def elapsed(self):
        return time.time() - self.start
            
            
            
            
        












"""
Class that holds 2D data 
"""
class CudaArray2D:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, stream, nx, ny, x_halo, y_halo, cpu_data=None, dtype=np.float32):
        self.logger =  logging.getLogger(__name__)
        self.nx = nx
        self.ny = ny
        self.x_halo = x_halo
        self.y_halo = y_halo
        
        nx_halo = nx + 2*x_halo
        ny_halo = ny + 2*y_halo
        
        #self.logger.debug("Allocating [%dx%d] buffer", self.nx, self.ny)
        #Should perhaps use pycuda.driver.mem_alloc_data.pitch() here
        self.data = pycuda.gpuarray.empty((ny_halo, nx_halo), dtype)
        
        #If we don't have any data, just allocate and return
        if cpu_data is None:
            return
            
        #Make sure data is in proper format
        assert cpu_data.shape == (ny_halo, nx_halo) or cpu_data.shape == (self.ny, self.nx), "Wrong shape of data %s vs %s / %s" % (str(cpu_data.shape), str((self.ny, self.nx)), str((ny_halo, nx_halo)))
        assert cpu_data.itemsize == 4, "Wrong size of data type"
        assert not np.isfortran(cpu_data), "Wrong datatype (Fortran, expected C)"

        #Create copy object from host to device
        copy = cuda.Memcpy2D()
        copy.set_src_host(cpu_data)
        copy.set_dst_device(self.data.gpudata)
            
        #Set offsets of upload in destination
        x_offset = (nx_halo - cpu_data.shape[1]) // 2
        y_offset = (ny_halo - cpu_data.shape[0]) // 2
        copy.dst_x_in_bytes = x_offset*self.data.strides[1]
        copy.dst_y = y_offset
        
        #Set destination pitch
        copy.dst_pitch = self.data.strides[0]
        
        #Set width in bytes to copy for each row and
        #number of rows to copy
        width = max(self.nx, cpu_data.shape[1])
        height = max(self.ny, cpu_data.shape[0])
        copy.width_in_bytes = width*cpu_data.itemsize
        copy.height = height
        
        #Perform the copy
        copy(stream)
        
        #self.logger.debug("Buffer <%s> [%dx%d]: Allocated ", int(self.data.gpudata), self.nx, self.ny)
        
        
    def __del__(self, *args):
        #self.logger.debug("Buffer <%s> [%dx%d]: Releasing ", int(self.data.gpudata), self.nx, self.ny)
        self.data.gpudata.free()
        self.data = None
        
    """
    Enables downloading data from GPU to Python
    """
    def download(self, stream, async=False):
        #self.logger.debug("Downloading [%dx%d] buffer", self.nx, self.ny)
        #Allocate host memory
        #cpu_data = cuda.pagelocked_empty((self.ny, self.nx), np.float32)
        cpu_data = np.empty((self.ny, self.nx), dtype=np.float32)
        
        #Create copy object from device to host
        copy = cuda.Memcpy2D()
        copy.set_src_device(self.data.gpudata)
        copy.set_dst_host(cpu_data)
        
        #Set offsets and pitch of source
        copy.src_x_in_bytes = self.x_halo*self.data.strides[1]
        copy.src_y = self.y_halo
        copy.src_pitch = self.data.strides[0]
        
        #Set width in bytes to copy for each row and
        #number of rows to copy
        copy.width_in_bytes = self.nx*cpu_data.itemsize
        copy.height = self.ny
        
        copy(stream)
        if async==False:
            stream.synchronize()
        
        return cpu_data

        
        
        
        
        
        
        
"""
Class that holds 2D data 
"""
class CudaArray3D:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, stream, nx, ny, nz, x_halo, y_halo, z_halo, cpu_data=None, dtype=np.float32):
        self.logger =  logging.getLogger(__name__)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x_halo = x_halo
        self.y_halo = y_halo
        self.z_halo = z_halo
        
        nx_halo = nx + 2*x_halo
        ny_halo = ny + 2*y_halo
        nz_halo = nz + 2*z_halo
        
        #self.logger.debug("Allocating [%dx%dx%d] buffer", self.nx, self.ny, self.nz)
        #Should perhaps use pycuda.driver.mem_alloc_data.pitch() here
        self.data = pycuda.gpuarray.empty((nz_halo, ny_halo, nx_halo), dtype)
        
        #If we don't have any data, just allocate and return
        if cpu_data is None:
            return
            
        #Make sure data is in proper format
        assert cpu_data.shape == (nz_halo, ny_halo, nx_halo) or cpu_data.shape == (self.nz, self.ny, self.nx), "Wrong shape of data %s vs %s / %s" % (str(cpu_data.shape), str((self.nz, self.ny, self.nx)), str((nz_halo, ny_halo, nx_halo)))
        assert cpu_data.itemsize == 4, "Wrong size of data type"
        assert not np.isfortran(cpu_data), "Wrong datatype (Fortran, expected C)"
            
        #Create copy object from host to device
        copy = cuda.Memcpy3D()
        copy.set_src_host(cpu_data)
        copy.set_dst_device(self.data.gpudata)
        
        #Set offsets of destination
        x_offset = (nx_halo - cpu_data.shape[2]) // 2
        y_offset = (ny_halo - cpu_data.shape[1]) // 2
        z_offset = (nz_halo - cpu_data.shape[0]) // 2
        copy.dst_x_in_bytes = x_offset*self.data.strides[1]
        copy.dst_y = y_offset
        copy.dst_z = z_offset
        
        #Set pitch of destination
        copy.dst_pitch = self.data.strides[0]
        
        #Set width in bytes to copy for each row and
        #number of rows to copy
        width = max(self.nx, cpu_data.shape[2])
        height = max(self.ny, cpu_data.shape[1])
        depth = max(self.nz, cpu-data.shape[0])
        copy.width_in_bytes = width*cpu_data.itemsize
        copy.height = height
        copy.depth = depth
        
        #Perform the copy
        copy(stream)
        
        #self.logger.debug("Buffer <%s> [%dx%d]: Allocated ", int(self.data.gpudata), self.nx, self.ny)
        
        
    def __del__(self, *args):
        #self.logger.debug("Buffer <%s> [%dx%d]: Releasing ", int(self.data.gpudata), self.nx, self.ny)
        self.data.gpudata.free()
        self.data = None
        
    """
    Enables downloading data from GPU to Python
    """
    def download(self, stream, async=False):
        #self.logger.debug("Downloading [%dx%d] buffer", self.nx, self.ny)
        #Allocate host memory
        #cpu_data = cuda.pagelocked_empty((self.ny, self.nx), np.float32)
        cpu_data = np.empty((self.nz, self.ny, self.nx), dtype=np.float32)
        
        #Create copy object from device to host
        copy = cuda.Memcpy2D()
        copy.set_src_device(self.data.gpudata)
        copy.set_dst_host(cpu_data)
        
        #Set offsets and pitch of source
        copy.src_x_in_bytes = self.x_halo*self.data.strides[1]
        copy.src_y = self.y_halo
        copy.src_z = self.z_halo
        copy.src_pitch = self.data.strides[0]
        
        #Set width in bytes to copy for each row and
        #number of rows to copy
        copy.width_in_bytes = self.nx*cpu_data.itemsize
        copy.height = self.ny
        copy.depth = self.nz
        
        copy(stream)
        if async==False:
            stream.synchronize()
        
        return cpu_data

        
        
        
        
        
        
        
        
"""
A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
"""
class ArakawaA2D:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, stream, nx, ny, halo_x, halo_y, cpu_variables):
        self.logger =  logging.getLogger(__name__)
        self.gpu_variables = []
        for cpu_variable in cpu_variables:
            self.gpu_variables += [CudaArray2D(stream, nx, ny, halo_x, halo_y, cpu_variable)]
        
    def __getitem__(self, key):
        assert type(key) == int, "Indexing is int based"
        if (key > len(self.gpu_variables) or key < 0):
            raise IndexError("Out of bounds")
        return self.gpu_variables[key]
    
    """
    Enables downloading data from CL device to Python
    """
    def download(self, stream):
        cpu_variables = []
        for gpu_variable in self.gpu_variables:
            cpu_variables += [gpu_variable.download(stream, async=True)]
        stream.synchronize()
        return cpu_variables
        
        
    
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

from GPUSimulators import Autotuner

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
            
            
            
            

"""
Class which keeps track of the CUDA context and some helper functions
"""
class CudaContext(object):
    
    def __init__(self, blocking=False, use_cache=True, autotuning=True):
        self.blocking = blocking
        self.use_cache = use_cache
        self.logger =  logging.getLogger(__name__)
        self.kernels = {}
        
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        
        #Initialize cuda (must be first call to PyCUDA)
        cuda.init(flags=0)
        
        self.logger.info("PyCUDA version %s", str(pycuda.VERSION_TEXT))
        
        #Print some info about CUDA
        self.logger.info("CUDA version %s", str(cuda.get_version()))
        self.logger.info("Driver version %s",  str(cuda.get_driver_version()))

        self.cuda_device = cuda.Device(0)
        self.logger.info("Using '%s' GPU", self.cuda_device.name())
        self.logger.debug(" => compute capability: %s", str(self.cuda_device.compute_capability()))

        # Create the CUDA context
        if (self.blocking):
            self.cuda_context = self.cuda_device.make_context(flags=cuda.ctx_flags.SCHED_BLOCKING_SYNC)
            self.logger.warning("Using blocking context")
        else:
            self.cuda_context = self.cuda_device.make_context(flags=cuda.ctx_flags.SCHED_AUTO)
            
        free, total = cuda.mem_get_info()
        self.logger.debug(" => memory: %d / %d MB available", int(free/(1024*1024)), int(total/(1024*1024)))
        
        self.logger.info("Created context handle <%s>", str(self.cuda_context.handle))
        
        #Create cache dir for cubin files
        if (self.use_cache):
            self.cache_path = os.path.join(self.module_path, "cuda_cache") 
            if not os.path.isdir(self.cache_path):
                os.mkdir(self.cache_path)
            self.logger.info("Using CUDA cache dir %s", self.cache_path)
            
        self.autotuner = None
        if (autotuning):
            self.logger.info("Autotuning enabled. It may take several minutes to run the code the first time: have patience")
            self.autotuner = Autotuner.Autotuner()
            
    
    def __del__(self, *args):
        self.logger.info("Cleaning up CUDA context handle <%s>", str(self.cuda_context.handle))
            
        # Loop over all contexts in stack, and remove "this"
        other_contexts = []
        while (cuda.Context.get_current() != None):
            context = cuda.Context.get_current()
            if (context.handle != self.cuda_context.handle):
                self.logger.debug("<%s> Popping <%s> (*not* ours)", str(self.cuda_context.handle), str(context.handle))
                other_contexts = [context] + other_contexts
                cuda.Context.pop()
            else:
                self.logger.debug("<%s> Popping <%s> (ours)", str(self.cuda_context.handle), str(context.handle))
                cuda.Context.pop()

        # Add all the contexts we popped that were not our own
        for context in other_contexts:
            self.logger.debug("<%s> Pushing <%s>", str(self.cuda_context.handle), str(context.handle))
            cuda.Context.push(context)
            
        self.logger.debug("<%s> Detaching", str(self.cuda_context.handle))
        self.cuda_context.detach()
        
        
    def __str__(self):
        return "CudaContext id " + str(self.cuda_context.handle)
        
    
    def hash_kernel(kernel_filename, include_dirs):        
        # Generate a kernel ID for our caches
        num_includes = 0
        max_includes = 100
        kernel_hasher = hashlib.md5()
        logger = logging.getLogger(__name__)
        
        # Loop over file and includes, and check if something has changed
        files = [kernel_filename]
        while len(files):
        
            if (num_includes > max_includes):
                raise("Maximum number of includes reached - circular include in {:}?".format(kernel_filename))
        
            filename = files.pop()
            
            #logger.debug("Hashing %s", filename)
                
            modified = os.path.getmtime(filename)
                
            # Open the file
            with io.open(filename, "r") as file:
            
                # Search for #inclue <something> and also hash the file
                file_str = file.read()
                kernel_hasher.update(file_str.encode('utf-8'))
                kernel_hasher.update(str(modified).encode('utf-8'))
                
                #Find all includes
                includes = re.findall('^\W*#include\W+(.+?)\W*$', file_str, re.M)
                
            # Loop over everything that looks like an include
            for include_file in includes:
                
                #Search through include directories for the file
                file_path = os.path.dirname(filename)
                for include_path in [file_path] + include_dirs:
                
                    # If we find it, add it to list of files to check
                    temp_path = os.path.join(include_path, include_file)
                    if (os.path.isfile(temp_path)):
                        files = files + [temp_path]
                        num_includes = num_includes + 1 #For circular includes...
                        break
            
        return kernel_hasher.hexdigest()
    
    """
    Reads a text file and creates an OpenCL kernel from that
    """
    def get_prepared_kernel(self, kernel_filename, kernel_function_name, \
                    prepared_call_args, \
                    include_dirs=[], no_extern_c=True, 
                    **kwargs):
        """
        Helper function to print compilation output
        """
        def cuda_compile_message_handler(compile_success_bool, info_str, error_str):
            self.logger.debug("Compilation returned %s", str(compile_success_bool))
            if info_str:
                self.logger.debug("Info: %s", info_str)
            if error_str:
                self.logger.debug("Error: %s", error_str)
        
        #self.logger.debug("Getting %s", kernel_filename)
            
        # Create a hash of the kernel (and its includes)
        kwargs_hasher = hashlib.md5()
        kwargs_hasher.update(str(kwargs).encode('utf-8'));
        kwargs_hash = kwargs_hasher.hexdigest()
        kwargs_hasher = None
        root, ext = os.path.splitext(kernel_filename)
        kernel_hash = root \
                + "_" + CudaContext.hash_kernel( \
                    os.path.join(self.module_path, kernel_filename), \
                    include_dirs=[self.module_path] + include_dirs) \
                + "_" + kwargs_hash \
                + ext
        cached_kernel_filename = os.path.join(self.cache_path, kernel_hash)
        
        # If we have the kernel in our hashmap, return it
        if (kernel_hash in self.kernels.keys()):
            self.logger.debug("Found kernel %s cached in hashmap (%s)", kernel_filename, kernel_hash)
            return self.kernels[kernel_hash]
        
        # If we have it on disk, return it
        elif (self.use_cache and os.path.isfile(cached_kernel_filename)):
            self.logger.debug("Found kernel %s cached on disk (%s)", kernel_filename, kernel_hash)
                
            with io.open(cached_kernel_filename, "rb") as file:
                file_str = file.read()
                module = cuda.module_from_buffer(file_str, message_handler=cuda_compile_message_handler)
                
            kernel = module.get_function(kernel_function_name)
            kernel.prepare(prepared_call_args)
            self.kernels[kernel_hash] = kernel
            return kernel
            
        # Otherwise, compile it from source
        else:
            self.logger.debug("Compiling %s (%s)", kernel_filename, kernel_hash)
                
            #Create kernel string
            kernel_string = ""
            for key, value in kwargs.items():
                kernel_string += "#define {:s} {:s}\n".format(str(key), str(value))
            kernel_string += '#include "{:s}"'.format(os.path.join(self.module_path, kernel_filename))
            if (self.use_cache):
                with io.open(cached_kernel_filename + ".txt", "w") as file:
                    file.write(kernel_string)
                
            
            with Timer("compiler") as timer:
                cubin = cuda_compiler.compile(kernel_string, include_dirs=include_dirs, no_extern_c=no_extern_c, cache_dir=False)
                module = cuda.module_from_buffer(cubin, message_handler=cuda_compile_message_handler)
                if (self.use_cache):
                    with io.open(cached_kernel_filename, "wb") as file:
                        file.write(cubin)
                
            kernel = module.get_function(kernel_function_name)
            kernel.prepare(prepared_call_args)
            self.kernels[kernel_hash] = kernel
            
            
            return kernel
    
    """
    Clears the kernel cache (useful for debugging & development)
    """
    def clear_kernel_cache(self):
        self.logger.debug("Clearing cache")
        self.kernels = {}
        gc.collect()
        
    """
    Synchronizes all streams etc
    """
    def synchronize(self):
        self.cuda_context.synchronize()
        












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
        
        
    
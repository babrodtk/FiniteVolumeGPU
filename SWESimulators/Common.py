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

import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda



"""
Class which keeps track of time spent for a section of code
"""
class Timer(object):
    def __init__(self, tag, verbose=True):
        self.verbose = verbose
        self.tag = tag
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000 # millisecs
        if self.verbose:
            print("=> " + self.tag + ' %f ms' % self.msecs)
            
            
            
            

"""
Class which keeps track of the CUDA context and some helper functions
"""
class CudaContext(object):
    
    def __init__(self, verbose=True, blocking=False, use_cache=True):
        self.verbose = verbose
        self.blocking = blocking
        self.use_cache = use_cache
        self.modules = {}
        
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        
        #Initialize cuda (must be first call to PyCUDA)
        cuda.init(flags=0)
        
        #Print some info about CUDA
        if (self.verbose):
            print("CUDA version " + str(cuda.get_version()))
            print("Driver version " + str(cuda.get_driver_version()))

        self.cuda_device = cuda.Device(0)
        if (self.verbose):
            print("Using " + self.cuda_device.name())
            print(" => compute capability: " + str(self.cuda_device.compute_capability()))
            print(" => memory: " + str(self.cuda_device.total_memory() / (1024*1024)) + " MB")

        # Create the CUDA context
        if (self.blocking):
            self.cuda_context = self.cuda_device.make_context(flags=cuda.ctx_flags.SCHED_BLOCKING_SYNC)
            if (self.verbose):
                print("=== WARNING ===")
                print("Using blocking context")
                print("=== WARNING ===")
        else:
            self.cuda_context = self.cuda_device.make_context(flags=cuda.ctx_flags.SCHED_AUTO)
        
        if (self.verbose):
            print("Created context <" + str(self.cuda_context.handle) + ">")
        
        #Create cache dir for cubin files
        if (self.use_cache):
            self.cache_path = os.path.join(self.module_path, "cuda_cache") 
            if not os.path.isdir(self.cache_path):
                os.mkdir(self.cache_path)
            if (verbose):
                print("Using CUDA cache dir " + self.cache_path)
            
    
    def __del__(self, *args):
        if self.verbose:
            print("Cleaning up CUDA context <" + str(self.cuda_context.handle) + ">")
            
        # Loop over all contexts in stack, and remove "this"
        other_contexts = []
        while (cuda.Context.get_current() != None):
            context = cuda.Context.get_current()
            if (self.verbose):
                if (context.handle != self.cuda_context.handle):
                    print(" `-> <" + str(self.cuda_context.handle) + "> Popping context <" + str(context.handle) + "> which we do not own")
                    other_contexts = [context] + other_contexts
                    cuda.Context.pop()
                else:
                    print(" `-> <" + str(self.cuda_context.handle) + "> Popping context <" + str(context.handle) + "> (ourselves)")
                    cuda.Context.pop()

        # Add all the contexts we popped that were not our own
        for context in other_contexts:
            if (self.verbose):
                print(" `-> <" + str(self.cuda_context.handle) + "> Pushing <" + str(context.handle) + ">")
            cuda.Context.push(context)
            
        if (self.verbose):
            print(" `-> <" + str(self.cuda_context.handle) + "> Detaching context")
        self.cuda_context.detach()
        
        
    def __str__(self):
        return "CudaContext id " + str(self.cuda_context.handle)
        
    
    def hash_kernel(kernel_filename, include_dirs, verbose=False):        
        # Generate a kernel ID for our caches
        num_includes = 0
        max_includes = 100
        kernel_hasher = hashlib.md5()
        
        with Timer("compiler", verbose=False) as timer:
            # Loop over file and includes, and check if something has changed
            files = [kernel_filename]
            while len(files):
            
                if (num_includes > max_includes):
                    raise("Maximum number of includes reached - circular include in {:}?".format(kernel_filename))
            
                filename = files.pop()
                
                if (verbose):
                    print("`-> Hashing " + filename)
                    
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
                
        if (verbose):
            print("`-> Hashed in " + str(timer.secs) + " seconds")
            
        return kernel_hasher.hexdigest()
    
    """
    Reads a text file and creates an OpenCL kernel from that
    """
    def get_prepared_kernel(self, kernel_filename, kernel_function_name, \
                    prepared_call_args, \
                    include_dirs=[], verbose=False, no_extern_c=False, 
                    **kwargs):
        """
        Helper function to print compilation output
        """
        def cuda_compile_message_handler(compile_success_bool, info_str, error_str):
            if (verbose):
                print("`-> Compilation returned " + str(compile_success_bool))
                if info_str:
                    print("`-> Info: " + info_str) 
                if error_str:
                    print("`-> Error: " + error_str) 
        
        if (verbose):
            print("Getting " + kernel_filename)
            
        # Create a hash of the kernel (and its includes)
        root, ext = os.path.splitext(kernel_filename)
        kernel_hash = root \
                + "_" + CudaContext.hash_kernel( \
                    os.path.join(self.module_path, kernel_filename), \
                    include_dirs=[self.module_path] + include_dirs, \
                    verbose=verbose) \
                + "_" + str(hash(str(kwargs))) \
                + ext
        cached_kernel_filename = os.path.join(self.cache_path, kernel_hash)
        
        # If we have the kernel in our hashmap, return it
        if (kernel_hash in self.modules.keys()):
            if (verbose):
                print("`-> Found kernel " + kernel_hash + " cached in hashmap")
            return self.modules[kernel_hash].get_function(kernel_function_name)
        
        # If we have it on disk, return it
        elif (self.use_cache and os.path.isfile(cached_kernel_filename)):
            if (verbose):
                print("`-> Found kernel " + kernel_hash + " cached on disk")
                
            with io.open(cached_kernel_filename, "rb") as file:
                file_str = file.read()
                module = cuda.module_from_buffer(file_str, message_handler=cuda_compile_message_handler)
                
            self.modules[kernel_hash] = module
            kernel = self.modules[kernel_hash].get_function(kernel_function_name)
            kernel.prepare(prepared_call_args)
            return kernel
            
        # Otherwise, compile it from source
        else:
            if (verbose):
                print("`-> Compiling " + kernel_filename)
                
            #Create kernel string
            kernel_string = ""
            for key, value in kwargs.items():
                kernel_string += "#define {:s} {:s}\n".format(str(key), str(value))
            kernel_string += '#include "' + os.path.join(self.module_path, kernel_filename) + '"'
            if (self.use_cache):
                with io.open(cached_kernel_filename + ".txt", "w") as file:
                    file.write(kernel_string)
                
            
            with Timer("compiler", verbose=False) as timer:
                cubin = cuda_compiler.compile(kernel_string, include_dirs=include_dirs, no_extern_c=no_extern_c, cache_dir=False)
                module = cuda.module_from_buffer(cubin, message_handler=cuda_compile_message_handler)
                self.modules[kernel_hash] = module
                if (self.use_cache):
                    with io.open(cached_kernel_filename, "wb") as file:
                        file.write(cubin)
                
            if (verbose):
                print("`-> Compiled in " + str(timer.secs) + " seconds")
            
            kernel = self.modules[kernel_hash].get_function(kernel_function_name)
            kernel.prepare(prepared_call_args)
            return kernel
    
    """
    Clears the kernel cache (useful for debugging & development)
    """
    def clear_kernel_cache(self):
        self.modules = {}
        
        
        
        
        
        

"""
Class that holds data 
"""
class CUDAArray2D:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, stream, nx, ny, halo_x, halo_y, data):
        
        self.nx = nx
        self.ny = ny
        self.nx_halo = nx + 2*halo_x
        self.ny_halo = ny + 2*halo_y
        
        #Make sure data is in proper format
        assert np.issubdtype(data.dtype, np.float32), "Wrong datatype: %s" % str(data.dtype)
        assert not np.isfortran(data), "Wrong datatype (Fortran, expected C)"
        assert data.shape == (self.ny_halo, self.nx_halo), "Wrong data shape: %s vs %s" % (str(data.shape), str((self.ny_halo, self.nx_halo)))

        #Upload data to the device
        self.data = pycuda.gpuarray.to_gpu_async(data, stream=stream)
        
        self.bytes_per_float = data.itemsize
        assert(self.bytes_per_float == 4)
        self.pitch = np.int32((self.nx_halo)*self.bytes_per_float)
        
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, stream, async=False):
        #Copy data from device to host
        if (async):
            host_data = self.data.get_async(stream=stream)
            return host_data
        else:
            host_data = self.data.get(stream=stream)#, pagelocked=True) # pagelocked causes crash on windows at least
            return host_data

        
        
        
        
        
        
        
        
"""
A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
"""
class SWEDataArakawaA:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, stream, nx, ny, halo_x, halo_y, h0, hu0, hv0):
        self.h0  = CUDAArray2D(stream, nx, ny, halo_x, halo_y, h0)
        self.hu0 = CUDAArray2D(stream, nx, ny, halo_x, halo_y, hu0)
        self.hv0 = CUDAArray2D(stream, nx, ny, halo_x, halo_y, hv0)
        
        self.h1  = CUDAArray2D(stream, nx, ny, halo_x, halo_y, h0)
        self.hu1 = CUDAArray2D(stream, nx, ny, halo_x, halo_y, hu0)
        self.hv1 = CUDAArray2D(stream, nx, ny, halo_x, halo_y, hv0)

    """
    Swaps the variables after a timestep has been completed
    """
    def swap(self):
        self.h1,  self.h0  = self.h0,  self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, stream):
        h_cpu  = self.h0.download(stream, async=True)
        hu_cpu = self.hu0.download(stream, async=True)
        hv_cpu = self.hv0.download(stream, async=False)
        
        return h_cpu, hu_cpu, hv_cpu
        
        

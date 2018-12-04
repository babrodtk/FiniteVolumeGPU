# -*- coding: utf-8 -*-

"""
This python module implements Cuda context handling

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

from GPUSimulators import Autotuner, Common



"""
Class which keeps track of the CUDA context and some helper functions
"""
class CudaContext(object):
    
    def __init__(self, device=None, context_flags=None, use_cache=True, autotuning=True):
        """
        Create a new CUDA context
        Set device to an id or pci_bus_id to select a specific GPU
        Set context_flags to cuda.ctx_flags.SCHED_BLOCKING_SYNC for a blocking context
        """
        self.use_cache = use_cache
        self.logger =  logging.getLogger(__name__)
        self.modules = {}
        
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        
        #Initialize cuda (must be first call to PyCUDA)
        cuda.init(flags=0)
        
        self.logger.info("PyCUDA version %s", str(pycuda.VERSION_TEXT))
        
        #Print some info about CUDA
        self.logger.info("CUDA version %s", str(cuda.get_version()))
        self.logger.info("Driver version %s",  str(cuda.get_driver_version()))
        
        if device is None:
            device = 0
        
        self.cuda_device = cuda.Device(device)
        self.logger.info("Using device %d/%d '%s' (%s) GPU", device, cuda.Device.count(), self.cuda_device.name(), self.cuda_device.pci_bus_id())
        self.logger.debug(" => compute capability: %s", str(self.cuda_device.compute_capability()))

        # Create the CUDA context
        if context_flags is None:
            context_flags=cuda.ctx_flags.SCHED_AUTO
            
        self.cuda_context = self.cuda_device.make_context(flags=context_flags)
            
        free, total = cuda.mem_get_info()
        self.logger.debug(" => memory: %d / %d MB available", int(free/(1024*1024)), int(total/(1024*1024)))
        
        self.logger.info("Created context handle <%s>", str(self.cuda_context.handle))
        
        #Create cache dir for cubin files
        self.cache_path = os.path.join(self.module_path, "cuda_cache") 
        if (self.use_cache):
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
    def get_module(self, kernel_filename, 
                    include_dirs=[], \
                    defines={}, \
                    compile_args={'no_extern_c', True}, jit_compile_args={}):
        """
        Helper function to print compilation output
        """
        def cuda_compile_message_handler(compile_success_bool, info_str, error_str):
            self.logger.debug("Compilation returned %s", str(compile_success_bool))
            if info_str:
                self.logger.debug("Info: %s", info_str)
            if error_str:
                self.logger.debug("Error: %s", error_str)
        
        kernel_filename = os.path.normpath(kernel_filename)
        kernel_path = os.path.abspath(os.path.join(self.module_path, kernel_filename))
        #self.logger.debug("Getting %s", kernel_filename)
            
        # Create a hash of the kernel options
        options_hasher = hashlib.md5()
        options_hasher.update(str(defines).encode('utf-8') + str(compile_args).encode('utf-8'));
        options_hash = options_hasher.hexdigest()
        
        # Create hash of kernel souce
        source_hash = CudaContext.hash_kernel( \
                    kernel_path, \
                    include_dirs=[self.module_path] + include_dirs)
                    
        # Create final hash
        root, ext = os.path.splitext(kernel_filename)
        kernel_hash = root \
                + "_" + source_hash \
                + "_" + options_hash \
                + ext
        cached_kernel_filename = os.path.join(self.cache_path, kernel_hash)
        
        # If we have the kernel in our hashmap, return it
        if (kernel_hash in self.modules.keys()):
            self.logger.debug("Found kernel %s cached in hashmap (%s)", kernel_filename, kernel_hash)
            return self.modules[kernel_hash]
        
        # If we have it on disk, return it
        elif (self.use_cache and os.path.isfile(cached_kernel_filename)):
            self.logger.debug("Found kernel %s cached on disk (%s)", kernel_filename, kernel_hash)
                
            with io.open(cached_kernel_filename, "rb") as file:
                file_str = file.read()
                module = cuda.module_from_buffer(file_str, message_handler=cuda_compile_message_handler, **jit_compile_args)
                
            self.modules[kernel_hash] = module
            return module
            
        # Otherwise, compile it from source
        else:
            self.logger.debug("Compiling %s (%s)", kernel_filename, kernel_hash)
                
            #Create kernel string
            kernel_string = ""
            for key, value in defines.items():
                kernel_string += "#define {:s} {:s}\n".format(str(key), str(value))
            kernel_string += '#include "{:s}"'.format(os.path.join(self.module_path, kernel_filename))
            if (self.use_cache):
                cached_kernel_dir = os.path.dirname(cached_kernel_filename)
                if not os.path.isdir(cached_kernel_dir):
                    os.mkdir(cached_kernel_dir)
                with io.open(cached_kernel_filename + ".txt", "w") as file:
                    file.write(kernel_string)
                
            
            with Common.Timer("compiler") as timer:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="The CUDA compiler succeeded, but said the following:\nkernel.cu", category=UserWarning)
                    cubin = cuda_compiler.compile(kernel_string, include_dirs=include_dirs, cache_dir=False, **compile_args)
                module = cuda.module_from_buffer(cubin, message_handler=cuda_compile_message_handler, **jit_compile_args)
                if (self.use_cache):
                    with io.open(cached_kernel_filename, "wb") as file:
                        file.write(cubin)
                
            self.modules[kernel_hash] = module
            return module
    
    """
    Clears the kernel cache (useful for debugging & development)
    """
    def clear_kernel_cache(self):
        self.logger.debug("Clearing cache")
        self.modules = {}
        gc.collect()
        
    """
    Synchronizes all streams etc
    """
    def synchronize(self):
        self.cuda_context.synchronize()
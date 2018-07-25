import os

import numpy as np
import time

import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda

"""
Class which keeps track of the CUDA context and some helper functions
"""
class CudaContext(object):
    def __init__(self, verbose=True, blocking=False):
        self.verbose = verbose
        self.blocking = blocking
        self.kernels = {}
        
        cuda.init(flags=0)
        
        if (self.verbose):
            print("CUDA version " + str(cuda.get_version()))
            print("Driver version " + str(cuda.get_driver_version()))

        self.cuda_device = cuda.Device(0)
        if (self.verbose):
            print("Using " + self.cuda_device.name())
            print(" => compute capability: " + str(self.cuda_device.compute_capability()))
            print(" => memory: " + str(self.cuda_device.total_memory() / (1024*1024)) + " MB")

        if (self.blocking):
            self.cuda_context = self.cuda_device.make_context(flags=cuda.ctx_flags.SCHED_BLOCKING_SYNC)
            if (self.verbose):
                print("=== WARNING ===")
                print("Using blocking context")
                print("=== WARNING ===")
        else:
            self.cuda_context = self.cuda_device.make_context(flags=cuda.ctx_flags.SCHED_AUTO)
            
    
    def __del__(self, *args):
        if self.verbose:
            print("Cleaning up CUDA context")
            
        self.cuda_context.detach()
        cuda.Context.pop()

            
    """
    Reads a text file and creates an OpenCL kernel from that
    """
    def get_kernel(self, kernel_filename, block_width, block_height):
        # Generate a kernel ID for our cache
        module_path = os.path.dirname(os.path.realpath(__file__))
        fullpath = os.path.join(module_path, kernel_filename)
        kernel_date = os.path.getmtime(fullpath)
        with open(fullpath, "r") as kernel_file:
            kernel_hash = hash(kernel_file.read())
        kernel_id = kernel_filename + ":" + str(kernel_hash) + ":" + str(kernel_date)
    
        # Simple caching to keep keep from recompiling kernels
        if (kernel_id not in self.kernels.keys()):
            #Create define string
            define_string = "#define block_width " + str(block_width) + "\n"
            define_string += "#define block_height " + str(block_height) + "\n\n"
            
            
            kernel_string = define_string + '#include "' + fullpath + '"'
            self.kernels[kernel_id] = cuda_compiler.SourceModule(kernel_string, include_dirs=[module_path])
            
        return self.kernels[kernel_id]
    
    """
    Clears the kernel cache (useful for debugging & development)
    """
    def clear_kernel_cache(self):
        self.kernels = {}
        
        
        
        
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
        assert data.shape == (self.ny_halo, self.nx_halo), "Wrong data shape: %s" % str(data.shape)

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
        
        

import os

import numpy as np

import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda

"""
Static function which reads a text file and creates an OpenCL kernel from that
"""
def get_kernel(kernel_filename, block_width, block_height):
    import datetime
    
    #Create define string
    define_string = "#define block_width " + str(block_width) + "\n"
    define_string += "#define block_height " + str(block_height) + "\n\n"
    
    
    #Read the proper program
    module_path = os.path.dirname(os.path.realpath(__file__))
    fullpath = os.path.join(module_path, kernel_filename)
    #with open(fullpath, "r") as kernel_file:
    #    kernel_string = define_string + kernel_file.read()
    #    kernel = cuda_compiler.SourceModule(kernel_string, include_dirs=[module_path], no_extern_c=True)
    
    kernel_string = define_string + '#include "' + fullpath + '"'
    kernel = cuda_compiler.SourceModule(kernel_string, include_dirs=[module_path])
        
    return kernel
    
    
        
        
        
        
        
        
        

"""
Class that holds data 
"""
class CUDAArray2D:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, nx, ny, halo_x, halo_y, data, stream=None):
        host_data = self.convert_to_float32(data)
        
        self.nx = nx
        self.ny = ny
        self.nx_halo = nx + 2*halo_x
        self.ny_halo = ny + 2*halo_y
        assert(host_data.shape[1] == self.nx_halo)
        assert(host_data.shape[0] == self.ny_halo)
        
        assert(data.shape == (self.ny_halo, self.nx_halo))

        #Upload data to the device
        self.data = pycuda.gpuarray.to_gpu_async(host_data, stream=stream)
        
        self.bytes_per_float = host_data.itemsize
        assert(self.bytes_per_float == 4)
        self.pitch = np.int32((self.nx_halo)*self.bytes_per_float)
        
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, stream=None, async=False):
        #Allocate data on the host for result
        
        #Copy data from device to host
        if (async):
            host_data = self.data.get_async(stream=stream)
            return host_data
        else:
            host_data = self.data.get(stream=stream)#, pagelocked=True) # pagelocked causes crash on windows at least
            return host_data

    """
    Converts to C-style float 32 array suitable for the GPU/OpenCL
    """
    @staticmethod
    def convert_to_float32(data):
        if (not np.issubdtype(data.dtype, np.float32) or np.isfortran(data)):
            print("WARNING: Converting DATA IN COMMON.PY")
            return data.astype(np.float32, order='C')
        else:
            return data

        
        
        
        
        
        
        
        
"""
A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
"""
class SWEDataArakawaA:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, nx, ny, halo_x, halo_y, h0, hu0, hv0, stream=None):
        self.h0  = CUDAArray2D(nx, ny, halo_x, halo_y, h0, stream=stream)
        self.hu0 = CUDAArray2D(nx, ny, halo_x, halo_y, hu0, stream=stream)
        self.hv0 = CUDAArray2D(nx, ny, halo_x, halo_y, hv0, stream=stream)
        
        self.h1  = CUDAArray2D(nx, ny, halo_x, halo_y, h0, stream=stream)
        self.hu1 = CUDAArray2D(nx, ny, halo_x, halo_y, hu0, stream=stream)
        self.hv1 = CUDAArray2D(nx, ny, halo_x, halo_y, hv0, stream=stream)

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
    def download(self, stream=None):
        h_cpu  = self.h0.download(stream=stream, async=True)
        hu_cpu = self.hu0.download(stream=stream, async=True)
        hv_cpu = self.hv0.download(stream=stream, async=False)
        
        return h_cpu, hu_cpu, hv_cpu
        
        
        
        
        
        
        
        

        
        
        
        
"""
A class representing an Akrawa C type (staggered, u fluxes on east/west faces, v fluxes on north/south faces) grid
We use h as cell centers
"""
class SWEDataArakawaC:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, nx, ny, halo_x, halo_y, h0, hu0, hv0):
        #FIXME: This at least works for 0 and 1 ghost cells, but not convinced it generalizes
        assert(halo_x <= 1 and halo_y <= 1)
        
        self.h0   = CUDAArray2D(nx, ny, halo_x, halo_y, h0)
        self.hu0  = CUDAArray2D(nx+1, ny, 0, halo_y, hu0)
        self.hv0  = CUDAArray2D(nx, ny+1, halo_x, 0, hv0)
        
        self.h1   = CUDAArray2D(nx, ny, halo_x, halo_y, h0)
        self.hu1  = CUDAArray2D(nx+1, ny, 0, halo_y, hu0)
        self.hv1  = CUDAArray2D(nx, ny+1, halo_x, 0, hv0)

    """
    Swaps the variables after a timestep has been completed
    """
    def swap(self):
        #h is assumed to be constant (bottom topography really)
        self.h1,  self.h0  = self.h0, self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, stream=None):
        h_cpu  = self.h0.download(stream=stream, async=True)
        hu_cpu = self.hu0.download(stream=stream, async=True)
        hv_cpu = self.hv0.download(stream=stream, async=False)
        
        return h_cpu, hu_cpu, hv_cpu
        
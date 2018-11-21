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
import subprocess
import tempfile
import re
import io
import hashlib
import logging
import gc
import netCDF4

import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda


class Timer(object):
    """
    Class which keeps track of time spent for a section of code
    """
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
            
            
            
            

class PopenFileBuffer(object):
    """
    Simple class for holding a set of tempfiles
    for communicating with a subprocess
    """
    def __init__(self):
        self.stdout = tempfile.TemporaryFile(mode='w+t')
        self.stderr = tempfile.TemporaryFile(mode='w+t')

    def __del__(self):
        self.stdout.close()
        self.stderr.close()

    def read(self):
        self.stdout.seek(0)
        cout = self.stdout.read()
        self.stdout.seek(0, 2)

        self.stderr.seek(0)
        cerr = self.stderr.read()
        self.stderr.seek(0, 2)

        return cout, cerr
            
            
            
            
class IPEngine(object):
    """
    Class for starting IPEngines for MPI processing in IPython
    """
    def __init__(self, n_engines):
        self.logger = logging.getLogger(__name__)
        
        #Start ipcontroller
        self.logger.info("Starting IPController")
        self.c_buff = PopenFileBuffer()
        c_cmd = ["ipcontroller",  "--ip='*'"]
        self.c = subprocess.Popen(c_cmd, stderr=self.c_buff.stderr, stdout=self.c_buff.stdout, shell=False)
        
        #Wait until controller is running
        time.sleep(3)
        
        #Start engines
        self.logger.info("Starting IPEngines")
        self.e_buff = PopenFileBuffer()
        e_cmd = ["mpiexec", "-n", str(n_engines), "ipengine", "--mpi"]
        self.e = subprocess.Popen(e_cmd, stderr=self.e_buff.stderr, stdout=self.e_buff.stdout, shell=False)

        # attach to a running cluster
        import ipyparallel
        self.cluster = ipyparallel.Client()#profile='mpi')
        while(len(self.cluster.ids) != n_engines):
            time.sleep(0.5)
            self.logger.info("Waiting for cluster...")
            self.cluster = ipyparallel.Client()#profile='mpi')
        
        self.logger.info("Done")
        
    def __del__(self):
        self.cluster.shutdown(hub=True)
        
        self.e.terminate()
        try:
            self.e.communicate(timeout=3)
        except TimeoutExpired:
            self.logger.warn("Killing IPEngine")
            self.e.kill()
            self.e.communicate()
            
        cout, cerr = self.e_buff.read()
        self.logger.info("IPEngine cout: {:s}".format(cout))
        self.logger.info("IPEngine cerr: {:s}".format(cerr))
        
        
        while(len(self.cluster.ids) != 0):
            time.sleep(0.5)
            self.logger.info("Waiting for cluster to shutdown...")
            
        
        self.c.terminate()
        try:
            self.c.communicate(timeout=3)
        except TimeoutExpired:
            self.logger.warn("Killing IPController")
            self.c.kill()
            self.c.communicate()
            
        cout, cerr = self.c_buff.read()
        self.logger.info("IPController cout: {:s}".format(cout))
        self.logger.info("IPController cerr: {:s}".format(cerr))

            
        


class DataDumper(object):
    """
    Simple class for holding a netCDF4 object
    (handles opening and closing in a nice way)
    Use as 
    with DataDumper("filename") as data:
        ...
    """
    def __init__(self, filename, *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        
        #Create directory if needed
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            self.logger.info("Creating directory " + dirname)
            os.makedirs(dirname)
        
        #Get mode of file if we have that
        mode = None
        if (args):
            mode = args[0]
        elif (kwargs and 'mode' in kwargs.keys()):
            mode = kwargs['mode']
            
        #Create new unique file if writing
        if (mode):
            if (("w" in mode) or ("+" in mode) or ("a" in mode)):
                i = 0
                stem, ext = os.path.splitext(filename)
                while (os.path.isfile(filename)):
                    filename = "{:s}_{:04d}{:s}".format(stem, i, ext)
                    i = i+1
        self.filename = os.path.abspath(filename)
        
        #Save arguments
        self.args = args
        self.kwargs = kwargs
                
        #Log output
        self.logger.info("Writing output to " + self.filename)
        
        
    def __enter__(self):
        self.logger.info("Opening " + self.filename)
        if (self.args):
            self.logger.info("Arguments: " + str(self.args))
        if (self.kwargs):
            self.logger.info("Keyword arguments: " + str(self.kwargs))
        self.ncfile = netCDF4.Dataset(self.filename, *self.args, **self.kwargs)
        return self
        
    def __exit__(self, *args):
        self.logger.info("Closing " + self.filename)
        self.ncfile.close()


        
        
class ProgressPrinter(object):
    """
    Small helper class for 
    """
    def __init__(self, total_steps, print_every=5):
        self.logger = logging.getLogger(__name__)
        self.start = time.time()
        self.total_steps = total_steps
        self.print_every = print_every
        self.next_print_time = self.print_every
        self.last_step = 0
        self.secs_per_iter = None
        
    def getPrintString(self, step):
        elapsed =  time.time() - self.start
        if (elapsed > self.next_print_time):            
            dt = elapsed - (self.next_print_time - self.print_every)
            dsteps = step - self.last_step
            steps_remaining = self.total_steps - step
                        
            if (dsteps == 0):
                return
                
            self.last_step = step
            self.next_print_time = elapsed + self.print_every
            
            if not self.secs_per_iter:
                self.secs_per_iter = dt / dsteps
            self.secs_per_iter = 0.2*self.secs_per_iter + 0.8*(dt / dsteps)
            
            remaining_time = steps_remaining * self.secs_per_iter

            return "{:s}. Total: {:s}, elapsed: {:s}, remaining: {:s}".format(
                ProgressPrinter.progressBar(step, self.total_steps), 
                ProgressPrinter.timeString(elapsed + remaining_time), 
                ProgressPrinter.timeString(elapsed), 
                ProgressPrinter.timeString(remaining_time))

    def timeString(seconds):
        seconds = int(max(seconds, 1))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        periods = [('h', hours), ('m', minutes), ('s', seconds)]
        time_string = ' '.join('{}{}'.format(value, name)
                                for name, value in periods
                                if value)
        return time_string

    def progressBar(step, total_steps, width=30):
        progress = np.round(width * step / total_steps).astype(np.int32)
        progressbar = "0% [" + "#"*(progress) + "="*(width-progress) + "] 100%"
        return progressbar







"""
Class that holds 2D data 
"""
class CudaArray2D:
    """
    Uploads initial data to the CUDA device
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
        self.data = pycuda.gpuarray.zeros((ny_halo, nx_halo), dtype)
        
        #If we don't have any data, just allocate and return
        if cpu_data is None:
            return
            
        #Make sure data is in proper format
        assert cpu_data.shape == (ny_halo, nx_halo) or cpu_data.shape == (self.ny, self.nx), "Wrong shape of data %s vs %s / %s" % (str(cpu_data.shape), str((self.ny, self.nx)), str((ny_halo, nx_halo)))
        assert cpu_data.itemsize == 4, "Wrong size of data type"
        assert not np.isfortran(cpu_data), "Wrong datatype (Fortran, expected C)"

        #Create copy object from host to device
        x = (nx_halo - cpu_data.shape[1]) // 2
        y = (ny_halo - cpu_data.shape[0]) // 2
        self.upload(stream, cpu_data, extent=[x, y, cpu_data.shape[1], cpu_data.shape[0]])
        #self.logger.debug("Buffer <%s> [%dx%d]: Allocated ", int(self.data.gpudata), self.nx, self.ny)
        
        
    def __del__(self, *args):
        #self.logger.debug("Buffer <%s> [%dx%d]: Releasing ", int(self.data.gpudata), self.nx, self.ny)
        self.data.gpudata.free()
        self.data = None
        
    """
    Enables downloading data from GPU to Python
    """
    def download(self, stream, cpu_data=None, async=False, extent=None):
        if (extent is None):
            x = self.x_halo
            y = self.y_halo
            nx = self.nx
            ny = self.ny
        else:
            x, y, nx, ny = extent
            
        if (cpu_data is None):
            #self.logger.debug("Downloading [%dx%d] buffer", self.nx, self.ny)
            #Allocate host memory
            #cpu_data = cuda.pagelocked_empty((self.ny, self.nx), np.float32)
            cpu_data = np.empty((ny, nx), dtype=np.float32)
            
        assert nx == cpu_data.shape[1]
        assert ny == cpu_data.shape[0]
        assert x+nx <= self.nx + 2*self.x_halo
        assert y+ny <= self.ny + 2*self.y_halo
        
        #Create copy object from device to host
        copy = cuda.Memcpy2D()
        copy.set_src_device(self.data.gpudata)
        copy.set_dst_host(cpu_data)
        
        #Set offsets and pitch of source
        copy.src_x_in_bytes = x*self.data.strides[1]
        copy.src_y = y
        copy.src_pitch = self.data.strides[0]
        
        #Set width in bytes to copy for each row and
        #number of rows to copy
        copy.width_in_bytes = nx*cpu_data.itemsize
        copy.height = ny
        
        copy(stream)
        if async==False:
            stream.synchronize()
        
        return cpu_data
        
        
    def upload(self, stream, cpu_data, extent=None):
        if (extent is None):
            x = self.x_halo
            y = self.y_halo
            nx = self.nx
            ny = self.ny
        else:
            x, y, nx, ny = extent
            
        assert(nx == cpu_data.shape[1])
        assert(ny == cpu_data.shape[0])
        assert(x+nx <= self.nx + 2*self.x_halo)
        assert(y+ny <= self.ny + 2*self.y_halo)
         
        #Create copy object from device to host
        copy = cuda.Memcpy2D()
        copy.set_dst_device(self.data.gpudata)
        copy.set_src_host(cpu_data)
        
        #Set offsets and pitch of source
        copy.dst_x_in_bytes = x*self.data.strides[1]
        copy.dst_y = y
        copy.dst_pitch = self.data.strides[0]
        
        #Set width in bytes to copy for each row and
        #number of rows to copy
        copy.width_in_bytes = nx*cpu_data.itemsize
        copy.height = ny
        
        copy(stream)

        
        
        
        
        
        
        
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
        
    """
    Checks that data is still sane
    """
    def check(self):
        for i, gpu_variable in enumerate(self.gpu_variables):
            var_sum = pycuda.gpuarray.sum(gpu_variable.data).get()
            self.logger.debug("Data %d with size [%d x %d] has sum %f", i, gpu_variable.nx, gpu_variable.ny, var_sum)
            assert np.isnan(var_sum) == False, "Data contains NaN values!"
    
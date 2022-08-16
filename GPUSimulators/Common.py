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
import signal
import subprocess
import tempfile
import re
import io
import hashlib
import logging
import gc
import netCDF4
import json

import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda
from pycuda.tools import PageLockedMemoryPool






def safeCall(cmd):
    logger = logging.getLogger(__name__)
    try:
        #git rev-parse HEAD
        current_dir = os.path.dirname(os.path.realpath(__file__))
        params = dict()
        params['stderr'] = subprocess.STDOUT
        params['cwd'] = current_dir
        params['universal_newlines'] = True #text=True in more recent python
        params['shell'] = False
        if os.name == 'nt':
            params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        stdout = subprocess.check_output(cmd, **params)
    except subprocess.CalledProcessError as e:
        output = e.output
        logger.error("Git failed, \nReturn code: " + str(e.returncode) + "\nOutput: " + output)
        raise e

    return stdout

def getGitHash():
    return safeCall(["git", "rev-parse", "HEAD"])

def getGitStatus():
    return safeCall(["git", "status", "--porcelain", "-uno"])

def toJson(in_dict, compressed=True):
    """
    Creates JSON string from a dictionary
    """
    logger = logging.getLogger(__name__)
    out_dict = in_dict.copy()
    for key in out_dict:
        if isinstance(out_dict[key], np.ndarray):
            out_dict[key] = out_dict[key].tolist()
        else:
            try:
                json.dumps(out_dict[key])
            except:
                value = str(out_dict[key])
                logger.warning("JSON: Converting {:s} to string ({:s})".format(key, value))
                out_dict[key] = value
    return json.dumps(out_dict)

def runSimulation(simulator, simulator_args, outfile, save_times, save_var_names=[], dt=None):
    """
    Runs a simulation, and stores output in netcdf file. Stores the times given in 
    save_times, and saves all of the variables in list save_var_names. Elements in  
    save_var_names can be set to None if you do not want to save them
    """
    profiling_data_sim_runner = { 'start': {}, 'end': {} }
    profiling_data_sim_runner["start"]["t_sim_init"] = 0
    profiling_data_sim_runner["end"]["t_sim_init"] = 0
    profiling_data_sim_runner["start"]["t_nc_write"] = 0
    profiling_data_sim_runner["end"]["t_nc_write"] = 0
    profiling_data_sim_runner["start"]["t_full_step"] = 0
    profiling_data_sim_runner["end"]["t_full_step"] = 0

    profiling_data_sim_runner["start"]["t_sim_init"] = time.time()

    logger = logging.getLogger(__name__)

    assert len(save_times) > 0, "Need to specify which times to save"

    with Timer("construct") as t:
        sim = simulator(**simulator_args)
    logger.info("Constructed in " + str(t.secs) + " seconds")

    #Create netcdf file and simulate
    with DataDumper(outfile, mode='w', clobber=False) as outdata:
        
        #Create attributes (metadata)
        outdata.ncfile.created = time.ctime(time.time())
        outdata.ncfile.git_hash = getGitHash()
        outdata.ncfile.git_status = getGitStatus()
        outdata.ncfile.simulator = str(simulator)
        
        # do not write fields to attributes (they are to large)
        simulator_args_for_ncfile = simulator_args.copy()
        del simulator_args_for_ncfile["rho"]
        del simulator_args_for_ncfile["rho_u"]
        del simulator_args_for_ncfile["rho_v"]
        del simulator_args_for_ncfile["E"]
        outdata.ncfile.sim_args = toJson(simulator_args_for_ncfile)
        
        #Create dimensions
        outdata.ncfile.createDimension('time', len(save_times))
        outdata.ncfile.createDimension('x', simulator_args['nx'])
        outdata.ncfile.createDimension('y', simulator_args['ny'])

        #Create variables for dimensions
        ncvars = {}
        ncvars['time'] = outdata.ncfile.createVariable('time', np.dtype('float32').char, 'time')
        ncvars['x']    = outdata.ncfile.createVariable(   'x', np.dtype('float32').char,    'x')
        ncvars['y']    = outdata.ncfile.createVariable(   'y', np.dtype('float32').char,    'y')
        
        #Fill variables with proper values
        ncvars['time'][:] = save_times
        extent = sim.getExtent()
        ncvars['x'][:] = np.linspace(extent[0], extent[1], simulator_args['nx'])
        ncvars['y'][:] = np.linspace(extent[2], extent[3], simulator_args['ny'])
        
        #Choose which variables to download (prune None from list, but keep the index)
        download_vars = []
        for i, var_name in enumerate(save_var_names):
            if var_name is not None:
                download_vars += [i]
        save_var_names = list(save_var_names[i] for i in download_vars)
        
        #Create variables
        for var_name in save_var_names:
            ncvars[var_name] = outdata.ncfile.createVariable(var_name, np.dtype('float32').char, ('time', 'y', 'x'), zlib=True, least_significant_digit=3)

        #Create step sizes between each save
        t_steps = np.empty_like(save_times)
        t_steps[0] = save_times[0]
        t_steps[1:] = save_times[1:] - save_times[0:-1]

        profiling_data_sim_runner["end"]["t_sim_init"] = time.time()

        #Start simulation loop
        progress_printer = ProgressPrinter(save_times[-1], print_every=10)
        for k in range(len(save_times)):
            #Get target time and step size there
            t_step = t_steps[k]
            t_end = save_times[k]
            
            #Sanity check simulator
            try:
                sim.check()
            except AssertionError as e:
                logger.error("Error after {:d} steps (t={:f}: {:s}".format(sim.simSteps(), sim.simTime(), str(e)))
                return outdata.filename

            profiling_data_sim_runner["start"]["t_full_step"] += time.time()

            #Simulate
            if (t_step > 0.0):
                sim.simulate(t_step, dt)

            profiling_data_sim_runner["end"]["t_full_step"] += time.time()

            profiling_data_sim_runner["start"]["t_nc_write"] += time.time()

            #Download
            save_vars = sim.download(download_vars)
            
            #Save to file
            for i, var_name in enumerate(save_var_names):
                ncvars[var_name][k, :] = save_vars[i]

            profiling_data_sim_runner["end"]["t_nc_write"] += time.time()

            #Write progress to screen
            print_string = progress_printer.getPrintString(t_end)
            if (print_string):
                logger.debug(print_string)
                
        logger.debug("Simulated to t={:f} in {:d} timesteps (average dt={:f})".format(t_end, sim.simSteps(), sim.simTime() / sim.simSteps()))

    return outdata.filename, profiling_data_sim_runner, sim.profiling_data_mpi






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
        c_params = dict()
        c_params['stderr'] = self.c_buff.stderr
        c_params['stdout'] = self.c_buff.stdout
        c_params['shell'] = False
        if os.name == 'nt':
            c_params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.c = subprocess.Popen(c_cmd, **c_params)
        
        #Wait until controller is running
        time.sleep(3)
        
        #Start engines
        self.logger.info("Starting IPEngines")
        self.e_buff = PopenFileBuffer()
        e_cmd = ["mpiexec", "-n", str(n_engines), "ipengine", "--mpi"]
        e_params = dict()
        e_params['stderr'] = self.e_buff.stderr
        e_params['stdout'] = self.e_buff.stdout
        e_params['shell'] = False
        if os.name == 'nt':
            e_params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.e = subprocess.Popen(e_cmd, **e_params)

        # attach to a running cluster
        import ipyparallel
        self.cluster = ipyparallel.Client()#profile='mpi')
        time.sleep(3)
        while(len(self.cluster.ids) != n_engines):
            time.sleep(0.5)
            self.logger.info("Waiting for cluster...")
            self.cluster = ipyparallel.Client()#profile='mpi')
        
        self.logger.info("Done")
        
    def __del__(self):
        self.shutdown()
    
    def shutdown(self):
        if (self.e is not None):
            if (os.name == 'nt'):
                self.logger.warn("Sending CTRL+C to IPEngine")
                self.e.send_signal(signal.CTRL_C_EVENT)
                
            try:
                self.e.communicate(timeout=3)
                self.e.kill()
            except subprocess.TimeoutExpired:
                self.logger.warn("Killing IPEngine")
                self.e.kill()
                self.e.communicate()
            self.e = None
                
            cout, cerr = self.e_buff.read()
            self.logger.info("IPEngine cout: {:s}".format(cout))
            self.logger.info("IPEngine cerr: {:s}".format(cerr))
            self.e_buff = None
            
            gc.collect()
            
        if (self.c is not None):
            if (os.name == 'nt'):
                self.logger.warn("Sending CTRL+C to IPController")
                self.c.send_signal(signal.CTRL_C_EVENT)
                
            try:
                self.c.communicate(timeout=3)
                self.c.kill()
            except subprocess.TimeoutExpired:
                self.logger.warn("Killing IPController")
                self.c.kill()
                self.c.communicate()
            self.c = None
                
            cout, cerr = self.c_buff.read()
            self.logger.info("IPController cout: {:s}".format(cout))
            self.logger.info("IPController cerr: {:s}".format(cerr))
            self.c_buff = None
        
            gc.collect()
        

            
        


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
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
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
        self.logger.info("Initialized " + self.filename)
        
        
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
        
        
    def toJson(in_dict):
        out_dict = in_dict.copy()

        for key in out_dict:
            if isinstance(out_dict[key], np.ndarray):
                out_dict[key] = out_dict[key].tolist()
            else:
                try:
                    json.dumps(out_dict[key])
                except:
                    out_dict[key] = str(out_dict[key])

        return json.dumps(out_dict)
        


        
        
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
        
        #For returning to download
        self.memorypool = PageLockedMemoryPool()
        
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
    def download(self, stream, cpu_data=None, asynch=False, extent=None):
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
            #The following fails, don't know why (crashes python)
            cpu_data = cuda.pagelocked_empty((int(ny), int(nx)), dtype=np.float32, mem_flags=cuda.host_alloc_flags.PORTABLE)
            #Non-pagelocked: cpu_data = np.empty((ny, nx), dtype=np.float32)
            #cpu_data = self.memorypool.allocate((ny, nx), dtype=np.float32)
            
        assert nx == cpu_data.shape[1]
        assert ny == cpu_data.shape[0]
        assert x+nx <= self.nx + 2*self.x_halo
        assert y+ny <= self.ny + 2*self.y_halo
        
        #Create copy object from device to host
        copy = cuda.Memcpy2D()
        copy.set_src_device(self.data.gpudata)
        copy.set_dst_host(cpu_data)
        
        #Set offsets and pitch of source
        copy.src_x_in_bytes = int(x)*self.data.strides[1]
        copy.src_y = int(y)
        copy.src_pitch = self.data.strides[0]
        
        #Set width in bytes to copy for each row and
        #number of rows to copy
        copy.width_in_bytes = int(nx)*cpu_data.itemsize
        copy.height = int(ny)
        
        copy(stream)
        if asynch==False:
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
        copy.dst_x_in_bytes = int(x)*self.data.strides[1]
        copy.dst_y = int(y)
        copy.dst_pitch = self.data.strides[0]
        
        #Set width in bytes to copy for each row and
        #number of rows to copy
        copy.width_in_bytes = int(nx)*cpu_data.itemsize
        copy.height = int(ny)
        
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
        self.data = pycuda.gpuarray.zeros((nz_halo, ny_halo, nx_halo), dtype)
        
        #For returning to download
        self.memorypool = PageLockedMemoryPool()
        
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
    def download(self, stream, asynch=False):
        #self.logger.debug("Downloading [%dx%d] buffer", self.nx, self.ny)
        #Allocate host memory
        #cpu_data = cuda.pagelocked_empty((self.ny, self.nx), np.float32)
        #cpu_data = np.empty((self.nz, self.ny, self.nx), dtype=np.float32)
        cpu_data = self.memorypool.allocate((self.nz, self.ny, self.nx), dtype=np.float32)
        
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
        if asynch==False:
            stream.synchronize()
        
        return cpu_data

        
        
        
        
        
        
        
        
"""
A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
"""
class ArakawaA2D:
    def __init__(self, stream, nx, ny, halo_x, halo_y, cpu_variables):
        """
        Uploads initial data to the GPU device
        """
        self.logger =  logging.getLogger(__name__)
        self.gpu_variables = []
        for cpu_variable in cpu_variables:
            self.gpu_variables += [CudaArray2D(stream, nx, ny, halo_x, halo_y, cpu_variable)]
        
    def __getitem__(self, key):
        assert type(key) == int, "Indexing is int based"
        if (key > len(self.gpu_variables) or key < 0):
            raise IndexError("Out of bounds")
        return self.gpu_variables[key]
    
    def download(self, stream, variables=None):
        """
        Enables downloading data from the GPU device to Python
        """
        if variables is None:
            variables=range(len(self.gpu_variables))
        
        cpu_variables = []
        for i in variables:
            assert i < len(self.gpu_variables), "Variable {:d} is out of range".format(i)
            cpu_variables += [self.gpu_variables[i].download(stream, asynch=True)]

        #stream.synchronize()
        return cpu_variables
        
    def check(self):
        """
        Checks that data is still sane
        """
        for i, gpu_variable in enumerate(self.gpu_variables):
            var_sum = pycuda.gpuarray.sum(gpu_variable.data).get()
            self.logger.debug("Data %d with size [%d x %d] has average %f", i, gpu_variable.nx, gpu_variable.ny, var_sum / (gpu_variable.nx * gpu_variable.ny))
            assert np.isnan(var_sum) == False, "Data contains NaN values!"
    
# -*- coding: utf-8 -*-

"""
This python module implements the classical Lax-Friedrichs numerical
scheme for the shallow water equations

Copyright (C) 2016  SINTEF ICT

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

#Import packages we need
import numpy as np
import logging

import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda

from SWESimulators import Common


class BaseSimulator:
    """
    Initialization routine
    context: GPU context to use
    kernel_wrapper: wrapper function of GPU kernel
    h0: Water depth incl ghost cells, (nx+1)*(ny+1) cells
    hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+1) cells
    hv0: Initial momentum along y-axis incl ghost cells, (nx+1)*(ny+1) cells
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    """
    def __init__(self, \
                 context, \
                 h0, hu0, hv0, \
                 nx, ny, \
                 ghost_cells_x, ghost_cells_y, \
                 dx, dy, dt, \
                 g, \
                 block_width, block_height):
        #Get logger
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        
        #Create a CUDA stream
        self.stream = cuda.Stream()
        
        #Create data by uploading to device
        self.data = Common.SWEDataArakawaA(self.stream, \
                            nx, ny, \
                            ghost_cells_x, ghost_cells_y, \
                            h0, hu0, hv0)
                           
        #Save input parameters
        #Notice that we need to specify them in the correct dataformat for the
        #GPU kernel
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.dt = np.float32(dt)
        self.g = np.float32(g) 
        
        #Keep track of simulation time
        self.t = 0.0;
                            
        #Compute kernel launch parameters
        self.local_size = (block_width, block_height, 1) 
        self.global_size = ( \
                       int(np.ceil(self.nx / float(self.local_size[0]))), \
                       int(np.ceil(self.ny / float(self.local_size[1]))) \
                      ) 

    """
    Function which simulates forward in time using the default simulation type
    """
    def simulate(self, t_end):
        raise(exceptions.NotImplementedError("Needs to be implemented in subclass"))
                      
    """ 
    Function which simulates t_end seconds using forward Euler
    Requires that the stepEuler functionality is implemented in the subclasses
    """
    def simulateEuler(self, t_end):
        with Common.Timer(self.__class__.__name__ + ".simulateEuler") as t:
            # Compute number of timesteps to perform
            n = int(t_end / self.dt + 1)
            
            for i in range(0, n):
                # Compute timestep for "this" iteration
                local_dt = np.float32(min(self.dt, t_end-i*self.dt))
                
                # Stop if end reached (should not happen)
                if (local_dt <= 0.0):
                    break
            
                # Step with forward Euler 
                self.stepEuler(local_dt)
            
        self.logger.info("%s simulated %f seconds to %f with %d steps in %f seconds", self.__class__.__name__, t_end, self.t, n, t.secs)
        return self.t, n
        
    """
    Function which simulates t_end seconds using Runge-Kutta 2
    Requires that the stepRK functionality is implemented in the subclasses
    """
    def simulateRK(self, t_end, order):
        with Common.Timer(self.__class__.__name__ + ".simulateRK") as t:
            # Compute number of timesteps to perform
            n = int(t_end / self.dt + 1)
            
            for i in range(0, n):
                # Compute timestep for "this" iteration
                local_dt = np.float32(min(self.dt, t_end-i*self.dt))
                
                # Stop if end reached (should not happen)
                if (local_dt <= 0.0):
                    break
            
                # Perform all the Runge-Kutta substeps
                self.stepRK(local_dt, order)
            
        self.logger.info("%s simulated %f seconds to %f with %d steps in %f seconds", self.__class__.__name__, t_end, self.t, n, t.secs)
        return self.t, n
        
    """
    Function which simulates t_end seconds using second order dimensional splitting (XYYX)
    Requires that the stepDimsplitX and stepDimsplitY functionality is implemented in the subclasses
    """
    def simulateDimsplit(self, t_end):
        with Common.Timer(self.__class__.__name__ + ".simulateDimsplit") as t:
            # Compute number of timesteps to perform
            n = int(t_end / (2.0*self.dt) + 1)
            
            for i in range(0, n):
                # Compute timestep for "this" iteration
                local_dt = np.float32(0.5*min(2*self.dt, t_end-2*i*self.dt))
                
                # Stop if end reached (should not happen)
                if (local_dt <= 0.0):
                    break
                
                # Perform the dimensional split substeps
                self.stepDimsplitXY(local_dt)
                self.stepDimsplitYX(local_dt)
            
        self.logger.info("%s simulated %f seconds to %f with %d steps in %f seconds", self.__class__.__name__, t_end, self.t, 2*n, t.secs)
        return self.t, 2*n
        
    
    """
    Function which performs one single timestep of size dt using forward euler
    """
    def stepEuler(self, dt):
        raise(NotImplementedError("Needs to be implemented in subclass"))
        
    def stepRK(self, dt, substep):
        raise(NotImplementedError("Needs to be implemented in subclass"))
    
    def stepDimsplitXY(self, dt):
        raise(NotImplementedError("Needs to be implemented in subclass"))
        
    def stepDimsplitYX(self, dt):
        raise(NotImplementedError("Needs to be implemented in subclass"))
        
    def sim_time(self):
        return self.t

    def download(self):
        return self.data.download(self.stream)


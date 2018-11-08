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
from enum import IntEnum

import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda

from GPUSimulators import Common


        


class BoundaryCondition(object):    
    """
    Class for holding boundary conditions for global boundaries
    """
    
    
    class Type(IntEnum):
        """
        Enum that describes the different types of boundary conditions
        WARNING: MUST MATCH THAT OF common.h IN CUDA
        """
        Dirichlet = 0,
        Neumann = 1,
        Periodic = 2,
        Reflective = 3
        
        
        
    def __init__(self, types={ \
                    'north': Type.Reflective, \
                    'south': Type.Reflective, \
                    'east': Type.Reflective, \
                    'west': Type.Reflective \
                 }):
        """
        Constructor
        """
        self.north = types['north']
        self.south = types['south']
        self.east = types['east']
        self.west = types['west']

        
    def asCodedInt(self):
        """
        Helper function which packs four boundary conditions into one integer
        """
        bc = 0
        bc = bc | (self.north & 0x000F) << 24
        bc = bc | (self.south & 0x000F) << 16
        bc = bc | (self.east & 0x000F) << 8
        bc = bc | (self.west & 0x000F) 
        
        #for t in types:
        #    print("{0:s}, {1:d}, {1:032b}, {1:08b}".format(t, types[t]))
        #print("bc: {0:032b}".format(bc))
        
        return np.int32(bc)    
    
    
    
    
class BaseSimulator(object):
   
    def __init__(self, \
                 context, \
                 nx, ny, \
                 dx, dy, dt, \
                 block_width, block_height):
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
        """
        #Get logger
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        
        #Save input parameters
        #Notice that we need to specify them in the correct dataformat for the
        #GPU kernel
        self.context = context
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.dt = np.float32(dt)
        
        #Handle autotuning block size
        if (self.context.autotuner):
            peak_configuration = self.context.autotuner.get_peak_performance(self.__class__)
            block_width = int(peak_configuration["block_width"])
            block_height = int(peak_configuration["block_height"])
            self.logger.debug("Used autotuning to get block size [%d x %d]", block_width, block_height)
        
        #Compute kernel launch parameters
        self.block_size = (block_width, block_height, 1) 
        self.grid_size = ( \
                       int(np.ceil(self.nx / float(self.block_size[0]))), \
                       int(np.ceil(self.ny / float(self.block_size[1]))) \
                      )
        
        #Create a CUDA stream
        self.stream = cuda.Stream()
        
        #Keep track of simulation time and number of timesteps
        self.t = 0.0
        self.nt = 0
        
                      
    def __str__(self):
        return "{:s} [{:d}x{:d}]".format(self.__class__.__name__, self.nx, self.ny)
                      
    
    def simulate(self, t_end):
        """
        Function which simulates forward in time using the default simulation type
        """
        raise(exceptions.NotImplementedError("Needs to be implemented in subclass"))
                      
    
    def simulateEuler(self, t_end):
        """ 
        Function which simulates t_end seconds using forward Euler
        Requires that the stepEuler functionality is implemented in the subclasses
        """
        # Compute number of timesteps to perform
        n = int(t_end / self.dt + 1)

        printer = Common.ProgressPrinter(n)
        
        for i in range(0, n):
            # Compute timestep for "this" iteration
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            
            # Stop if end reached (should not happen)
            if (local_dt <= 0.0):
                break
        
            # Step with forward Euler 
            self.stepEuler(local_dt)

            #Print info
            print_string = printer.getPrintString(i)
            if (print_string):
                self.logger.info("%s (Euler): %s", self, print_string)
                try:
                    self.check()
                except AssertionError as e:
                    e.args += ("Step={:d}, time={:f}".format(self.simSteps(), self.simTime()))
                    raise
    
            
        #self.logger.info("%s simulated %f seconds to %f with %d steps (Euler)", self, t_end, self.t, n)
        return self.t, n
        
    def simulateRK(self, t_end, order):    
        """
        Function which simulates t_end seconds using Runge-Kutta 2
        Requires that the stepRK functionality is implemented in the subclasses
        """
        # Compute number of timesteps to perform
        n = int(t_end / self.dt + 1)
        
        printer = Common.ProgressPrinter(n)
        
        for i in range(0, n):
            # Compute timestep for "this" iteration
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            
            # Stop if end reached (should not happen)
            if (local_dt <= 0.0):
                break
        
            # Perform all the Runge-Kutta substeps
            self.stepRK(local_dt, order)

            #Print info
            print_string = printer.getPrintString(i)
            if (print_string):
                self.logger.info("%s (RK2): %s", self, print_string)
                try:
                    self.check()
                except AssertionError as e:
                    e.args += ("Step={:d}, time={:f}".format(self.simSteps(), self.simTime()))
                    raise
    
        return self.t, n
        
    
    def simulateDimsplit(self, t_end):
        """
        Function which simulates t_end seconds using second order dimensional splitting (XYYX)
        Requires that the stepDimsplitX and stepDimsplitY functionality is implemented in the subclasses
        """
        # Compute number of timesteps to perform
        n = int(t_end / (2.0*self.dt) + 1)
        
        printer = Common.ProgressPrinter(n)

        for i in range(0, n):
            # Compute timestep for "this" iteration
            local_dt = np.float32(0.5*min(2*self.dt, t_end-2*i*self.dt))
            
            # Stop if end reached (should not happen)
            if (local_dt <= 0.0):
                break
            
            # Perform the dimensional split substeps
            self.stepDimsplitXY(local_dt)
            self.stepDimsplitYX(local_dt)

            #Print info
            print_string = printer.getPrintString(i)
            if (print_string):
                self.logger.info("%s (Dimsplit): %s", self, print_string)
                try:
                    self.check()
                except AssertionError as e:
                    e.args += ("Step={:d}, time={:f}".format(self.simSteps(), self.simTime()))
                    raise
            
        return self.t, 2*n
        
    
    def stepEuler(self, dt):
        """
        Function which performs one single timestep of size dt using forward euler
        """
        raise(NotImplementedError("Needs to be implemented in subclass"))
        
    def stepRK(self, dt, substep):
        """
        Function which performs one single timestep of size dt using Runge-Kutta
        """
        raise(NotImplementedError("Needs to be implemented in subclass"))
    
    def stepDimsplitXY(self, dt):
        """
        Function which performs one single timestep of size dt using dimensional splitting
        """
        raise(NotImplementedError("Needs to be implemented in subclass"))
        
    def stepDimsplitYX(self, dt):
        """
        Function which performs one single timestep of size dt using dimensional splitting
        """
        raise(NotImplementedError("Needs to be implemented in subclass"))

    def download(self):
        raise(NotImplementedError("Needs to be implemented in subclass"))
        
    def synchronize(self):
        self.stream.synchronize()

    def check(self):
        raise(NotImplementedError("Needs to be implemented in subclass"))
        
    def simTime(self):
        return self.t

    def simSteps(self):
        return self.nt
        
        
        
        
        
        
        
        
        
        
        
        
        
        
def stepOrderToCodedInt(step, order):
    """
    Helper function which packs the step and order into a single integer
    """
    step_order = (step << 16) ^ (order & 0x00ff)
    #print("Step:  {0:032b}".format(step))
    #print("Order: {0:032b}".format(order))
    #print("Mix:   {0:032b}".format(step_order))
    return np.int32(step_order)
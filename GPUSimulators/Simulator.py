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

    def __init__(self, types={ 
                    'north': Type.Reflective, 
                    'south': Type.Reflective, 
                    'east': Type.Reflective, 
                    'west': Type.Reflective 
                 }):
        """
        Constructor
        """
        self.north = types['north']
        self.south = types['south']
        self.east = types['east']
        self.west = types['west']
        
        if (self.north == BoundaryCondition.Type.Neumann \
                or self.south == BoundaryCondition.Type.Neumann \
                or self.east == BoundaryCondition.Type.Neumann \
                or self.west == BoundaryCondition.Type.Neumann):
            raise(NotImplementedError("Neumann boundary condition not supported"))
            
    def __str__(self):
        return  '[north={:s}, south={:s}, east={:s}, west={:s}]'.format(str(self.north), str(self.south), str(self.east), str(self.west))

        
    def asCodedInt(self):
        """
        Helper function which packs four boundary conditions into one integer
        """
        bc = 0
        bc = bc | (self.north & 0x0000000F) << 24
        bc = bc | (self.south & 0x0000000F) << 16
        bc = bc | (self.east  & 0x0000000F) <<  8
        bc = bc | (self.west  & 0x0000000F) <<  0
        
        #for t in types:
        #    print("{0:s}, {1:d}, {1:032b}, {1:08b}".format(t, types[t]))
        #print("bc: {0:032b}".format(bc))
        
        return np.int32(bc)
        
    def getTypes(bc):
        types = {}
        types['north'] = BoundaryCondition.Type((bc >> 24) & 0x0000000F)
        types['south'] = BoundaryCondition.Type((bc >> 16) & 0x0000000F)
        types['east']  = BoundaryCondition.Type((bc >>  8) & 0x0000000F)
        types['west']  = BoundaryCondition.Type((bc >>  0) & 0x0000000F)
        return types
        
    
    
    
    
    
    
    
class BaseSimulator(object):
   
    def __init__(self, 
                 context, 
                 nx, ny, 
                 dx, dy, 
                 boundary_conditions,
                 cfl_scale,
                 num_substeps,
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
        cfl_scale: Courant number
        num_substeps: Number of substeps to perform for a full step
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
        self.setBoundaryConditions(boundary_conditions)
        self.cfl_scale = cfl_scale
        self.num_substeps = num_substeps
        
        #Handle autotuning block size
        if (self.context.autotuner):
            peak_configuration = self.context.autotuner.get_peak_performance(self.__class__)
            block_width = int(peak_configuration["block_width"])
            block_height = int(peak_configuration["block_height"])
            self.logger.debug("Used autotuning to get block size [%d x %d]", block_width, block_height)
        
        #Compute kernel launch parameters
        self.block_size = (block_width, block_height, 1) 
        self.grid_size = ( 
                       int(np.ceil(self.nx / float(self.block_size[0]))), 
                       int(np.ceil(self.ny / float(self.block_size[1]))) 
                      )
        
        #Create a CUDA stream
        self.stream = cuda.Stream()
        self.internal_stream = cuda.Stream()
        
        #Keep track of simulation time and number of timesteps
        self.t = 0.0
        self.nt = 0
        

    def __str__(self):
        return "{:s} [{:d}x{:d}]".format(self.__class__.__name__, self.nx, self.ny)


    def simulate(self, t, dt=None):
        """ 
        Function which simulates t_end seconds using the step function
        Requires that the step() function is implemented in the subclasses
        """

        printer = Common.ProgressPrinter(t)
        
        t_start = self.simTime()
        t_end = t_start + t
        
        update_dt = True
        if (dt is not None):
            update_dt = False
            self.dt = dt
        
        while(self.simTime() < t_end):
            # Update dt every 100 timesteps and cross your fingers it works
            # for the next 100
            if (update_dt and (self.simSteps() % 100 == 0)):
                self.dt = self.computeDt()*self.cfl_scale
        
            # Compute timestep for "this" iteration (i.e., shorten last timestep)
            current_dt = np.float32(min(self.dt, t_end-self.simTime()))

            # Stop if end reached (should not happen)
            if (current_dt <= 0.0):
                self.logger.warning("Timestep size {:d} is less than or equal to zero!".format(self.simSteps()))
                break
        
            # Step forward in time
            self.step(current_dt)

            #Print info
            print_string = printer.getPrintString(self.simTime() - t_start)
            if (print_string):
                self.logger.info("%s: %s", self, print_string)
                try:
                    self.check()
                except AssertionError as e:
                    e.args += ("Step={:d}, time={:f}".format(self.simSteps(), self.simTime()),)
                    raise


    def step(self, dt):
        """
        Function which performs one single timestep of size dt
        """
        for i in range(self.num_substeps):
            self.substep(dt, i)
            
        self.t += dt
        self.nt += 1

    def download(self, variables=None):
        return self.getOutput().download(self.stream, variables)
        
    def synchronize(self):
        self.stream.synchronize()
        
    def simTime(self):
        return self.t

    def simSteps(self):
        return self.nt
       
    def getExtent(self):
        return [0, 0, self.nx*self.dx, self.ny*self.dy]
        
    def setBoundaryConditions(self, boundary_conditions):
        self.logger.debug("Boundary conditions set to {:s}".format(str(boundary_conditions)))
        self.boundary_conditions = boundary_conditions.asCodedInt()
        
    def getBoundaryConditions(self):
        return BoundaryCondition(BoundaryCondition.getTypes(self.boundary_conditions))
        
    def substep(self, dt, step_number):
        """
        Function which performs one single substep with stepsize dt
        """
        raise(NotImplementedError("Needs to be implemented in subclass"))
        
    def getOutput(self):
        raise(NotImplementedError("Needs to be implemented in subclass"))

    def check(self):
        self.logger.warning("check() is not implemented - please implement")
        #raise(NotImplementedError("Needs to be implemented in subclass"))
        
    def computeDt(self):
        raise(NotImplementedError("Needs to be implemented in subclass"))
       
        
        
        
        
        
        
        
        
        
        
        
        
def stepOrderToCodedInt(step, order):
    """
    Helper function which packs the step and order into a single integer
    """
    step_order = (step << 16) | (order & 0x0000ffff)
    #print("Step:  {0:032b}".format(step))
    #print("Order: {0:032b}".format(order))
    #print("Mix:   {0:032b}".format(step_order))
    return np.int32(step_order)
# -*- coding: utf-8 -*-

"""
This python module implements the 2nd order HLL flux

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
from SWESimulators import Simulator


        
        
        
        
        


"""
Class that solves the SW equations using the Forward-Backward linear scheme
"""
class HLL2 (Simulator.BaseSimulator):

    """
    Initialization routine
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
                 dx, dy, dt, \
                 g, \
                 theta=1.8, \
                 block_width=16, block_height=16):
                 
        # Call super constructor
        super().__init__(context, \
            h0, hu0, hv0, \
            nx, ny, \
            2, 2, \
            dx, dy, dt, \
            g, \
            block_width, block_height);
            
        self.theta = np.float32(theta)

        #Get kernels
        self.kernel = context.get_prepared_kernel("HLL2_kernel.cu", "HLL2Kernel", \
                                        "iifffffiPiPiPiPiPiPi", \
                                        BLOCK_WIDTH=block_width, \
                                        BLOCK_HEIGHT=block_height)
        
    def __str__(self):
        return "Harten-Lax-van Leer (2nd order)"
    
    def simulate(self, t_end):
        return super().simulateDimsplit(t_end)
        
    def stepEuler(self, dt):
        return self.stepDimsplitXY(dt)
                
    def stepDimsplitXY(self, dt):
        self.kernel.prepared_async_call(self.global_size, self.local_size, self.stream, \
                self.nx, self.ny, \
                self.dx, self.dy, dt, \
                self.g, \
                self.theta, \
                np.int32(0), \
                self.data.h0.data.gpudata, self.data.h0.pitch, \
                self.data.hu0.data.gpudata, self.data.hu0.pitch, \
                self.data.hv0.data.gpudata, self.data.hv0.pitch, \
                self.data.h1.data.gpudata, self.data.h1.pitch, \
                self.data.hu1.data.gpudata, self.data.hu1.pitch, \
                self.data.hv1.data.gpudata, self.data.hv1.pitch)
        self.data.swap()
        self.t += dt
            
    def stepDimsplitYX(self, dt):
        self.kernel.prepared_async_call(self.global_size, self.local_size, self.stream, \
                self.nx, self.ny, \
                self.dx, self.dy, dt, \
                self.g, \
                self.theta, \
                np.int32(1), \
                self.data.h0.data.gpudata, self.data.h0.pitch, \
                self.data.hu0.data.gpudata, self.data.hu0.pitch, \
                self.data.hv0.data.gpudata, self.data.hv0.pitch, \
                self.data.h1.data.gpudata, self.data.h1.pitch, \
                self.data.hu1.data.gpudata, self.data.hu1.pitch, \
                self.data.hv1.data.gpudata, self.data.hv1.pitch)
        self.data.swap()
        self.t += dt
        
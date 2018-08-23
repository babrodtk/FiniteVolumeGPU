# -*- coding: utf-8 -*-

"""
This python module implements the Kurganov-Petrova numerical scheme 
for the shallow water equations, described in 
A. Kurganov & Guergana Petrova
A Second-Order Well-Balanced Positivity Preserving Central-Upwind
Scheme for the Saint-Venant System Communications in Mathematical
Sciences, 5 (2007), 133-160. 

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
from GPUSimulators import Simulator





"""
Class that solves the SW equations using the Forward-Backward linear scheme
"""
class KP07 (Simulator.BaseSimulator):

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
                 theta=1.3, \
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
        self.kernel = context.get_prepared_kernel("KP07_kernel.cu", "KP07Kernel", \
                                        "iifffffiPiPiPiPiPiPi", \
                                        BLOCK_WIDTH=self.local_size[0], \
                                        BLOCK_HEIGHT=self.local_size[1])
        
    def __str__(self):
        return "Kurganov-Petrova 2007"
    
    def simulate(self, t_end):
        return super().simulateRK(t_end, 2)
        
    def substepRK(self, dt, substep):
        self.kernel.prepared_async_call(self.global_size, self.local_size, self.stream, \
                self.nx, self.ny, \
                self.dx, self.dy, dt, \
                self.g, \
                self.theta, \
                np.int32(substep), \
                self.data.h0.data.gpudata,  self.data.h0.data.strides[0],  \
                self.data.hu0.data.gpudata, self.data.hu0.data.strides[0], \
                self.data.hv0.data.gpudata, self.data.hv0.data.strides[0], \
                self.data.h1.data.gpudata,  self.data.h1.data.strides[0],  \
                self.data.hu1.data.gpudata, self.data.hu1.data.strides[0], \
                self.data.hv1.data.gpudata, self.data.hv1.data.strides[0])
        self.data.swap()
        
    def stepEuler(self, dt):
        self.substepRK(dt, 0)
        self.t += dt
        
    def stepRK(self, dt, order):
        if (order != 2):
            raise NotImplementedError("Only second order implemented")
        self.substepRK(dt, 0)
        self.substepRK(dt, 1)
        self.t += dt
    
    def download(self):
        return self.data.download(self.stream)


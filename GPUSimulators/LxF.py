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
from GPUSimulators import Simulator, Common, CudaContext







"""
Class that solves the SW equations using the Lax Friedrichs scheme
"""
class LxF (Simulator.BaseSimulator):

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
                 block_width=16, block_height=16):
                 
        # Call super constructor
        super().__init__(context, \
            nx, ny, \
            dx, dy, dt, \
            g, \
            block_width, block_height);

        # Get kernels
        self.kernel = context.get_prepared_kernel("cuda/SWE_LxF.cu", "LxFKernel", \
                                        "iiffffPiPiPiPiPiPi", \
                                        BLOCK_WIDTH=self.local_size[0], \
                                        BLOCK_HEIGHT=self.local_size[1])

        #Create data by uploading to device
        self.u0 = Common.ArakawaA2D(self.stream, \
                        nx, ny, \
                        1, 1, \
                        [h0, hu0, hv0])
        self.u1 = Common.ArakawaA2D(self.stream, \
                        nx, ny, \
                        1, 1, \
                        [None, None, None])
        
    def __str__(self):
        return "Lax Friedrichs"
        
    def simulate(self, t_end):
        return super().simulateEuler(t_end)
        
    def stepEuler(self, dt):
        self.kernel.prepared_async_call(self.global_size, self.local_size, self.stream, \
                self.nx, self.ny, \
                self.dx, self.dy, dt, \
                self.g, \
                self.u0[0].data.gpudata, self.u0[0].data.strides[0], \
                self.u0[1].data.gpudata, self.u0[1].data.strides[0], \
                self.u0[2].data.gpudata, self.u0[2].data.strides[0], \
                self.u1[0].data.gpudata, self.u1[0].data.strides[0], \
                self.u1[1].data.gpudata, self.u1[1].data.strides[0], \
                self.u1[2].data.gpudata, self.u1[2].data.strides[0])
        self.u0, self.u1 = self.u1, self.u0
        self.t += dt
  
    def download(self):
        return self.u0.download(self.stream)

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
from GPUSimulators import Simulator, Common
from GPUSimulators.Simulator import BaseSimulator, BoundaryCondition
import numpy as np

from pycuda import gpuarray


        
        
        
        
        


"""
Class that solves the SW equations using the Forward-Backward linear scheme
"""
class EE2D_KP07_dimsplit (BaseSimulator):

    """
    Initialization routine
    rho: Density
    rho_u: Momentum along x-axis
    rho_v: Momentum along y-axis
    E: energy
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis
    dy: Grid cell spacing along y-axis
    dt: Size of each timestep 
    g: Gravitational constant
    gamma: Gas constant
    p: pressure
    """
    def __init__(self, 
                 context, 
                 rho, rho_u, rho_v, E, 
                 nx, ny, 
                 dx, dy, dt, 
                 g, 
                 gamma, 
                 theta=1.3, 
                 cfl_scale=0.9,
                 boundary_conditions=BoundaryCondition(), 
                 block_width=16, block_height=8):
                 
        # Call super constructor
        super().__init__(context, \
            nx, ny, \
            dx, dy, 2*dt, \
            block_width, block_height)
        self.g = np.float32(g)
        self.gamma = np.float32(gamma)
        self.theta = np.float32(theta) 
        self.cfl_scale = cfl_scale
        self.boundary_conditions = boundary_conditions.asCodedInt()

        #Get kernels
        module = context.get_module("cuda/EE2D_KP07_dimsplit.cu", 
                                        defines={
                                            'BLOCK_WIDTH': self.block_size[0], 
                                            'BLOCK_HEIGHT': self.block_size[1]
                                        }, 
                                        compile_args={
                                            'no_extern_c': True,
                                            'options': ["--use_fast_math"], 
                                        }, 
                                        jit_compile_args={})
        self.kernel = module.get_function("KP07DimsplitKernel")
        self.kernel.prepare("iiffffffiiPiPiPiPiPiPiPiPiP")
        
        
        #Create data by uploading to device
        self.u0 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        2, 2, 
                        [rho, rho_u, rho_v, E])
        self.u1 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        2, 2, 
                        [None, None, None, None])
        self.cfl_data = gpuarray.GPUArray(self.grid_size, dtype=np.float32)
        self.cfl_data.fill(self.dt, stream=self.stream)
                        
    
    def step(self, dt):
        self.substepDimsplit(0.5*dt, 0)
        self.substepDimsplit(0.5*dt, 1)
        self.t += dt
        self.nt += 2
                
    def substepDimsplit(self, dt, substep):
        self.kernel.prepared_async_call(self.grid_size, self.block_size, self.stream, 
                self.nx, self.ny, 
                self.dx, self.dy, dt, 
                self.g, 
                self.gamma, 
                self.theta, 
                substep,
                self.boundary_conditions, 
                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
                self.u0[3].data.gpudata, self.u0[3].data.strides[0], 
                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
                self.u1[2].data.gpudata, self.u1[2].data.strides[0], 
                self.u1[3].data.gpudata, self.u1[3].data.strides[0],
                self.cfl_data.gpudata)
        self.u0, self.u1 = self.u1, self.u0
        
    def download(self):
        return self.u0.download(self.stream)

    def check(self):
        self.u0.check()
        self.u1.check()
        
    def computeDt(self):
        max_dt = gpuarray.min(self.cfl_data, stream=self.stream).get();
        return max_dt*0.5*self.cfl_scale
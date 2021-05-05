# -*- coding: utf-8 -*-

"""
This python module implements SHMEM simulator class

Copyright (C) 2020 Norwegian Meteorological Institute

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


import logging
from GPUSimulators import Simulator
import numpy as np


class SHMEMSimulator(Simulator.BaseSimulator):
    """
    Class which handles communication between simulators on different GPUs

    NOTE: This class is only intended to be used by SHMEMSimulatorGroup
    """
    def __init__(self, index, sim, grid):
        self.logger =  logging.getLogger(__name__)
        
        autotuner = sim.context.autotuner
        sim.context.autotuner = None;
        boundary_conditions = sim.getBoundaryConditions()
        super().__init__(sim.context, 
            sim.nx, sim.ny, 
            sim.dx, sim.dy, 
            boundary_conditions,
            sim.cfl_scale,
            sim.num_substeps,  
            sim.block_size[0], sim.block_size[1])
        sim.context.autotuner = autotuner
        
        self.index = index
        self.sim = sim
        self.grid = grid
        
        #Get neighbor subdomain ids
        self.east = grid.getEast(self.index)
        self.west = grid.getWest(self.index)
        self.north = grid.getNorth(self.index)
        self.south = grid.getSouth(self.index)
        
        #Get coordinate of this subdomain
        #and handle global boundary conditions
        new_boundary_conditions = Simulator.BoundaryCondition({
            'north': Simulator.BoundaryCondition.Type.Dirichlet,
            'south': Simulator.BoundaryCondition.Type.Dirichlet,
            'east': Simulator.BoundaryCondition.Type.Dirichlet,
            'west': Simulator.BoundaryCondition.Type.Dirichlet
        })
        gi, gj = grid.getCoordinate(self.index)
        if (gi == 0 and boundary_conditions.west != Simulator.BoundaryCondition.Type.Periodic):
            self.west = None
            new_boundary_conditions.west = boundary_conditions.west;
        if (gj == 0 and boundary_conditions.south != Simulator.BoundaryCondition.Type.Periodic):
            self.south = None
            new_boundary_conditions.south = boundary_conditions.south;
        if (gi == grid.grid[0]-1 and boundary_conditions.east != Simulator.BoundaryCondition.Type.Periodic):
            self.east = None
            new_boundary_conditions.east = boundary_conditions.east;
        if (gj == grid.grid[1]-1 and boundary_conditions.north != Simulator.BoundaryCondition.Type.Periodic):
            self.north = None
            new_boundary_conditions.north = boundary_conditions.north;
        sim.setBoundaryConditions(new_boundary_conditions)
                
        #Get number of variables
        self.nvars = len(self.getOutput().gpu_variables)
        
        #Shorthands for computing extents and sizes
        gc_x = int(self.sim.getOutput()[0].x_halo)
        gc_y = int(self.sim.getOutput()[0].y_halo)
        nx = int(self.sim.nx)
        ny = int(self.sim.ny)
        
        #Set regions for ghost cells to read from
        #These have the format [x0, y0, width, height]
        self.read_e = np.array([  nx,    0, gc_x, ny + 2*gc_y])
        self.read_w = np.array([gc_x,    0, gc_x, ny + 2*gc_y])
        self.read_n = np.array([gc_x,   ny,   nx,        gc_y])
        self.read_s = np.array([gc_x, gc_y,   nx,        gc_y])
        
        #Set regions for ghost cells to write to
        self.write_e = self.read_e + np.array([gc_x, 0, 0, 0])
        self.write_w = self.read_w - np.array([gc_x, 0, 0, 0])
        self.write_n = self.read_n + np.array([0, gc_y, 0, 0])
        self.write_s = self.read_s - np.array([0, gc_y, 0, 0])
        
        #Allocate data for receiving
        #Note that east and west also transfer ghost cells
        #whilst north/south only transfer internal cells
        #Reuses the width/height defined in the read-extets above
        self.in_e = np.empty((self.nvars, self.read_e[3], self.read_e[2]), dtype=np.float32)
        self.in_w = np.empty((self.nvars, self.read_w[3], self.read_w[2]), dtype=np.float32)
        self.in_n = np.empty((self.nvars, self.read_n[3], self.read_n[2]), dtype=np.float32)
        self.in_s = np.empty((self.nvars, self.read_s[3], self.read_s[2]), dtype=np.float32)
        
        #Allocate data for sending
        self.out_e = np.empty_like(self.in_e)
        self.out_w = np.empty_like(self.in_w)
        self.out_n = np.empty_like(self.in_n)
        self.out_s = np.empty_like(self.in_s)
        
        self.logger.debug("Simlator subdomain {:d} initialized on context {:s}".format(self.index, sim.context))
    
        
    def substep(self, dt, step_number):
        self.exchange()
        self.sim.substep(dt, step_number)
    
    def getOutput(self):
        return self.sim.getOutput()
        
    def synchronize(self):
        self.sim.synchronize()
        
    def check(self):
        return self.sim.check()
    
    def computeDt(self):
        local_dt = np.array([np.float32(self.sim.computeDt())])
        global_dt = np.empty(1, dtype=np.float32)
        self.grid.comm.Allreduce(local_dt, global_dt, op=MPI.MIN)
        self.logger.debug("Local dt: {:f}, global dt: {:f}".format(local_dt[0], global_dt[0]))
        return global_dt[0]
        
    def getExtent(self):
        """
        Function which returns the extent of the subdomain with index 
        index in the grid
        """
        width = self.sim.nx*self.sim.dx
        height = self.sim.ny*self.sim.dy
        i, j = self.grid.getCoordinate(self.index)
        x0 = i * width
        y0 = j * height 
        x1 = x0 + width
        y1 = y0 + height
        return [x0, x1, y0, y1]
        
    def exchange(self):        
        ns_download_before_exchange()
        # GLOBAL SYNC 
        ns_do_exchange()
        # GLOBAL SYNC
        ns_upload_after_exchange()

        ew_download_before_exchange()
        # GLOBAL SYNC
        ew_do_exchange()
        # GLOBAL SYNC
        ew_upload_after_exchange()

    def ns_download_before_exchange(self):
        ####
        # First transfer internal cells north-south
        ####
        
        #Download from the GPU
        if self.north is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_n[k,:,:], asynch=True, extent=self.read_n)
        if self.south is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_s[k,:,:], asynch=True, extent=self.read_s)
        self.sim.stream.synchronize()

    def ns_do_exchange(self):
        #Send/receive to north/south neighbours
        comm_send = []
        comm_recv = []
        if self.north is not None:
            comm_send += [self.grid.comm.Isend(self.out_n, dest=self.north, tag=4*self.nt + 0)]
            comm_recv += [self.grid.comm.Irecv(self.in_n, source=self.north, tag=4*self.nt + 1)]
        if self.south is not None:
            comm_send += [self.grid.comm.Isend(self.out_s, dest=self.south, tag=4*self.nt + 1)]
            comm_recv += [self.grid.comm.Irecv(self.in_s, source=self.south, tag=4*self.nt + 0)]
        
        #Wait for incoming transfers to complete
        for comm in comm_recv:
            comm.wait()

    def ns_upload_after_exchange(self):
        #Upload to the GPU
        if self.north is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_n[k,:,:], extent=self.write_n)
        if self.south is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_s[k,:,:], extent=self.write_s)
        
        #Wait for sending to complete
        for comm in comm_send:
            comm.wait()

    def ew_download_before_exchange(self):
        ####
        # Then transfer east-west including ghost cells that have been filled in by north-south transfer above
        ####
        
        #Download from the GPU
        if self.east is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_e[k,:,:], asynch=True, extent=self.read_e)
        if self.west is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_w[k,:,:], asynch=True, extent=self.read_w)
        self.sim.stream.synchronize()

    def ew_do_exchange(self):
        #Send/receive to east/west neighbours
        comm_send = []
        comm_recv = []
        if self.east is not None:
            comm_send += [self.grid.comm.Isend(self.out_e, dest=self.east, tag=4*self.nt + 2)]
            comm_recv += [self.grid.comm.Irecv(self.in_e, source=self.east, tag=4*self.nt + 3)]
        if self.west is not None:
            comm_send += [self.grid.comm.Isend(self.out_w, dest=self.west, tag=4*self.nt + 3)]
            comm_recv += [self.grid.comm.Irecv(self.in_w, source=self.west, tag=4*self.nt + 2)]
        
        
        #Wait for incoming transfers to complete
        for comm in comm_recv:
            comm.wait()

    def ew_upload_after_exchange(self):
        #Upload to the GPU
        if self.east is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_e[k,:,:], extent=self.write_e)
        if self.west is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_w[k,:,:], extent=self.write_w)
        
        #Wait for sending to complete
        for comm in comm_send:
            comm.wait()
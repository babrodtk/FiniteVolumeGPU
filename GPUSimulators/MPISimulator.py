# -*- coding: utf-8 -*-

"""
This python module implements MPI simulator class

Copyright (C) 2018 SINTEF Digital

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
from mpi4py import MPI
import time

import pycuda.driver as cuda
#import nvtx



class MPIGrid(object):
    """
    Class which represents an MPI grid of nodes. Facilitates easy communication between
    neighboring nodes
    """
    def __init__(self, comm, ndims=2):
        self.logger =  logging.getLogger(__name__)
        
        assert ndims == 2, "Unsupported number of dimensions. Must be two at the moment"
        assert comm.size >= 1, "Must have at least one node"
        
        self.grid = MPIGrid.getGrid(comm.size, ndims)
        self.comm = comm
        
        self.logger.debug("Created MPI grid: {:}. Rank {:d} has coordinate {:}".format(
                self.grid, self.comm.rank, self.getCoordinate()))

    def getCoordinate(self, rank=None):
        if (rank is None):
            rank = self.comm.rank
        i = (rank  % self.grid[0])
        j = (rank // self.grid[0])
        return i, j

    def getRank(self, i, j):
        return j*self.grid[0] + i

    def getEast(self):
        i, j = self.getCoordinate(self.comm.rank)
        i = (i+1) % self.grid[0]
        return self.getRank(i, j)

    def getWest(self):
        i, j = self.getCoordinate(self.comm.rank)
        i = (i+self.grid[0]-1) % self.grid[0]
        return self.getRank(i, j)

    def getNorth(self):
        i, j = self.getCoordinate(self.comm.rank)
        j = (j+1) % self.grid[1]
        return self.getRank(i, j)

    def getSouth(self):
        i, j = self.getCoordinate(self.comm.rank)
        j = (j+self.grid[1]-1) % self.grid[1]
        return self.getRank(i, j)
    
    def getGrid(num_nodes, num_dims):
        assert(isinstance(num_nodes, int))
        assert(isinstance(num_dims, int))
        
        # Adapted from https://stackoverflow.com/questions/28057307/factoring-a-number-into-roughly-equal-factors
        # Original code by https://stackoverflow.com/users/3928385/ishamael
        # Factorizes a number into n roughly equal factors

        #Dictionary to remember already computed permutations
        memo = {}
        def dp(n, left): # returns tuple (cost, [factors])
            """
            Recursively searches through all factorizations
            """

            #Already tried: return existing result
            if (n, left) in memo: 
                return memo[(n, left)]

            #Spent all factors: return number itself
            if left == 1:
                return (n, [n])

            #Find new factor
            i = 2
            best = n
            bestTuple = [n]
            while i * i < n:
                #If factor found
                if n % i == 0:
                    #Factorize remainder
                    rem = dp(n // i, left - 1)

                    #If new permutation better, save it
                    if rem[0] + i < best:
                        best = rem[0] + i
                        bestTuple = [i] + rem[1]
                i += 1

            #Store calculation
            memo[(n, left)] = (best, bestTuple)
            return memo[(n, left)]


        grid = dp(num_nodes, num_dims)[1]

        if (len(grid) < num_dims):
            #Split problematic 4
            if (4 in grid):
                grid.remove(4)
                grid.append(2)
                grid.append(2)

            #Pad with ones to guarantee num_dims
            grid = grid + [1]*(num_dims - len(grid))
        
        #Sort in descending order
        grid = np.sort(grid)
        grid = grid[::-1]

        # XXX: We only use vertical (north-south) partitioning for now
        grid[0] = 1
        grid[1] = num_nodes
        
        return grid


    def gather(self, data, root=0):
        out_data = None
        if (self.comm.rank == root):
            out_data = np.empty([self.comm.size] + list(data.shape), dtype=data.dtype)
        self.comm.Gather(data, out_data, root)
        return out_data
        
    def getLocalRank(self):
        """
        Returns the local rank on this node for this MPI process
        """
        
        # This function has been adapted from 
        # https://github.com/SheffieldML/PyDeepGP/blob/master/deepgp/util/parallel.py
        # by Zhenwen Dai released under BSD 3-Clause "New" or "Revised" License:
        # 
        # Copyright (c) 2016, Zhenwen Dai
        # All rights reserved.
        # 
        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions are met:
        # 
        # * Redistributions of source code must retain the above copyright notice, this
        #   list of conditions and the following disclaimer.
        # 
        # * Redistributions in binary form must reproduce the above copyright notice,
        #   this list of conditions and the following disclaimer in the documentation
        #   and/or other materials provided with the distribution.
        # 
        # * Neither the name of DGP nor the names of its
        #   contributors may be used to endorse or promote products derived from
        #   this software without specific prior written permission.
        # 
        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        
        #Get this ranks unique (physical) node name
        node_name = MPI.Get_processor_name()
        
        #Gather the list of all node names on all nodes
        node_names = self.comm.allgather(node_name)
                
        #Loop over all node names up until our rank
        #and count how many duplicates of our nodename we find
        local_rank = len([0 for name in node_names[:self.comm.rank] if name==node_name])
        
        return local_rank


class MPISimulator(Simulator.BaseSimulator):
    """
    Class which handles communication between simulators on different MPI nodes
    """
    def __init__(self, sim, grid):        
        self.profiling_data_mpi = { 'start': {}, 'end': {} }
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange"] = 0
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange"] = 0
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_download"] = 0
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_download"] = 0
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_upload"] = 0
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_upload"] = 0
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] = 0
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] = 0
        self.profiling_data_mpi["start"]["t_mpi_step"] = 0
        self.profiling_data_mpi["end"]["t_mpi_step"] = 0
        self.profiling_data_mpi["n_time_steps"] = 0
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
        
        self.sim = sim
        self.grid = grid
        
        #Get neighbor node ids
        self.east = grid.getEast()
        self.west = grid.getWest()
        self.north = grid.getNorth()
        self.south = grid.getSouth()
        
        #Get coordinate of this node
        #and handle global boundary conditions
        new_boundary_conditions = Simulator.BoundaryCondition({
            'north': Simulator.BoundaryCondition.Type.Dirichlet,
            'south': Simulator.BoundaryCondition.Type.Dirichlet,
            'east': Simulator.BoundaryCondition.Type.Dirichlet,
            'west': Simulator.BoundaryCondition.Type.Dirichlet
        })
        gi, gj = grid.getCoordinate()
        #print("gi: " + str(gi) + ", gj: " + str(gj))
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
        self.in_e = cuda.pagelocked_empty((int(self.nvars), int(self.read_e[3]), int(self.read_e[2])), dtype=np.float32) #np.empty((self.nvars, self.read_e[3], self.read_e[2]), dtype=np.float32)
        self.in_w = cuda.pagelocked_empty((int(self.nvars), int(self.read_w[3]), int(self.read_w[2])), dtype=np.float32) #np.empty((self.nvars, self.read_w[3], self.read_w[2]), dtype=np.float32)
        self.in_n = cuda.pagelocked_empty((int(self.nvars), int(self.read_n[3]), int(self.read_n[2])), dtype=np.float32) #np.empty((self.nvars, self.read_n[3], self.read_n[2]), dtype=np.float32)
        self.in_s = cuda.pagelocked_empty((int(self.nvars), int(self.read_s[3]), int(self.read_s[2])), dtype=np.float32) #np.empty((self.nvars, self.read_s[3], self.read_s[2]), dtype=np.float32)

        #Allocate data for sending
        self.out_e = cuda.pagelocked_empty((int(self.nvars), int(self.read_e[3]), int(self.read_e[2])), dtype=np.float32) #np.empty_like(self.in_e)
        self.out_w = cuda.pagelocked_empty((int(self.nvars), int(self.read_w[3]), int(self.read_w[2])), dtype=np.float32) #np.empty_like(self.in_w)
        self.out_n = cuda.pagelocked_empty((int(self.nvars), int(self.read_n[3]), int(self.read_n[2])), dtype=np.float32) #np.empty_like(self.in_n)
        self.out_s = cuda.pagelocked_empty((int(self.nvars), int(self.read_s[3]), int(self.read_s[2])), dtype=np.float32) #np.empty_like(self.in_s)
        
        self.logger.debug("Simlator rank {:d} initialized on {:s}".format(self.grid.comm.rank, MPI.Get_processor_name()))

        self.full_exchange()
        sim.context.synchronize()
    
    def substep(self, dt, step_number):
        
        #nvtx.mark("substep start", color="yellow")

        self.profiling_data_mpi["start"]["t_mpi_step"] += time.time()
        
        #nvtx.mark("substep external", color="blue")
        self.sim.substep(dt, step_number, external=True, internal=False) # only "internal ghost cells"
        
        #nvtx.mark("substep internal", color="red")
        self.sim.substep(dt, step_number, internal=True, external=False) # "internal ghost cells" excluded

        #nvtx.mark("substep full", color="blue")
        #self.sim.substep(dt, step_number, external=True, internal=True)

        self.sim.swapBuffers()

        self.profiling_data_mpi["end"]["t_mpi_step"] += time.time()
        
        #nvtx.mark("exchange", color="blue")
        self.full_exchange()

        #nvtx.mark("sync start", color="blue")
        self.sim.stream.synchronize()
        self.sim.internal_stream.synchronize()
        #nvtx.mark("sync end", color="blue")
        
        self.profiling_data_mpi["n_time_steps"] += 1

    def getOutput(self):
        return self.sim.getOutput()
        
    def synchronize(self):
        self.sim.synchronize()
        
    def check(self):
        return self.sim.check()
        
    def computeDt(self):
        local_dt = np.array([np.float32(self.sim.computeDt())]);
        global_dt = np.empty(1, dtype=np.float32)
        self.grid.comm.Allreduce(local_dt, global_dt, op=MPI.MIN)
        self.logger.debug("Local dt: {:f}, global dt: {:f}".format(local_dt[0], global_dt[0]))
        return global_dt[0]
        
        
    def getExtent(self):
        """
        Function which returns the extent of node with rank 
        rank in the grid
        """
        width = self.sim.nx*self.sim.dx
        height = self.sim.ny*self.sim.dy
        i, j = self.grid.getCoordinate()
        x0 = i * width
        y0 = j * height 
        x1 = x0 + width
        y1 = y0 + height
        return [x0, x1, y0, y1]

    def full_exchange(self):
        ####
        # First transfer internal cells north-south
        ####
        
        #Download from the GPU
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_download"] += time.time()
            
        if self.north is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_n[k,:,:], asynch=True, extent=self.read_n)
        if self.south is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_s[k,:,:], asynch=True, extent=self.read_s)
        self.sim.stream.synchronize()
        
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_download"] += time.time()
        
        #Send/receive to north/south neighbours
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] += time.time()
        
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
        
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] += time.time()
        
        #Upload to the GPU
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_upload"] += time.time()
        
        if self.north is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_n[k,:,:], extent=self.write_n)
        if self.south is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_s[k,:,:], extent=self.write_s)
                
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_upload"] += time.time()
        
        #Wait for sending to complete
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] += time.time()
        
        for comm in comm_send:
            comm.wait()
        
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] += time.time()
        
        ####
        # Then transfer east-west including ghost cells that have been filled in by north-south transfer above
        ####
        
        #Download from the GPU
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_download"] += time.time()
        
        if self.east is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_e[k,:,:], asynch=True, extent=self.read_e)
        if self.west is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_w[k,:,:], asynch=True, extent=self.read_w)
        self.sim.stream.synchronize()
        
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_download"] += time.time()
        
        #Send/receive to east/west neighbours
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] += time.time()
        
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
        
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] += time.time()
        
        #Upload to the GPU
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_upload"] += time.time()
        
        if self.east is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_e[k,:,:], extent=self.write_e)
        if self.west is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_w[k,:,:], extent=self.write_w)
        
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_upload"] += time.time()
        
        #Wait for sending to complete
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] += time.time()
        
        for comm in comm_send:
            comm.wait()
        
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] += time.time()

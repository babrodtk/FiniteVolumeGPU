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
        grid = np.flip(np.sort(grid))
        
        return grid
        
        
    def getExtent(self, width, height, rank):
        """
        Function which returns the extent of node with rank 
        rank in the grid
        """
        i, j = self.getCoordinate(rank)
        x0 = i * width
        y0 = j * height 
        x1 = x0+width
        y1 = y0+height
        return [x0, x1, y0, y1]


    def gatherData(self, data, rank=0):
        """
        Function which gathers the data onto node with rank
        rank
        """
        #Get shape of data
        ny, nx = data.shape
        
        #Create list of buffers to return
        retval = []
        
        #If we are the target node, recieve from others
        #otherwise send to target
        if (self.comm.rank == rank):
            mpi_requests = []
            retval = []
            
            #Loop over all nodes
            for k in range(0, self.comm.size):
                #If k equal target node, add our own data
                #Otherwise receive it from node k
                if (k == rank):
                    retval  += [data]
                else:
                    buffer = np.empty((ny, nx), dtype=np.float32)
                    retval += [buffer]
                    mpi_requests += [self.comm.Irecv(buffer, source=k, tag=k)]
                
            #Wait for transfers to complete
            for mpi_request in mpi_requests:
                mpi_request.wait()
        else:
            mpi_request = self.comm.Isend(data, dest=rank, tag=self.comm.rank)
            mpi_request.wait()
            
        return retval



class MPISimulator(Simulator.BaseSimulator):
    """
    Class which handles communication between simulators on different MPI nodes
    """
    def __init__(self, sim, grid):
        self.logger =  logging.getLogger(__name__)
        
        autotuner = sim.context.autotuner
        sim.context.autotuner = None;
        super().__init__(sim.context, 
            sim.nx, sim.ny, 
            sim.dx, sim.dy, 
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
        
        #Get number of variables
        self.nvars = len(self.sim.u0.gpu_variables)
        
        #Shorthands for computing extents and sizes
        gc_x = int(self.sim.u0[0].x_halo)
        gc_y = int(self.sim.u0[0].y_halo)
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
        
        self.logger.debug("Simlator rank {:d} initialized ".format(self.grid.comm.rank))
    
        
    def substep(self, dt, step_number):
        self.exchange()
        self.sim.substep(dt, step_number)
    
    def download(self):
        return self.sim.download()
        
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
        
    def exchange(self):        
        ####
        # FIXME: This function can be optimized using persitent communications. 
        # Also by overlapping some of the communications north/south and east/west of GPU and intra-node
        # communications
        ####
        
        ####
        # First transfer internal cells north-south
        ####
        
        #Download from the GPU
        for k in range(self.nvars):
            self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_n[k,:,:], async=True, extent=self.read_n)
            self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_s[k,:,:], async=True, extent=self.read_s)
        self.sim.stream.synchronize()
        
        #Send to north/south neighbours
        comm_send = []
        comm_send += [self.grid.comm.Isend(self.out_n, dest=self.north, tag=4*self.nt + 0)]
        comm_send += [self.grid.comm.Isend(self.out_s, dest=self.south, tag=4*self.nt + 1)]
        
        #Receive from north/south neighbors
        comm_recv = []
        comm_recv += [self.grid.comm.Irecv(self.in_s, source=self.south, tag=4*self.nt + 0)]
        comm_recv += [self.grid.comm.Irecv(self.in_n, source=self.north, tag=4*self.nt + 1)]
        
        #Wait for incoming transfers to complete
        for comm in comm_recv:
            comm.wait()
        
        #Upload to the GPU
        for k in range(self.nvars):
            self.sim.u0[k].upload(self.sim.stream, self.in_n[k,:,:], extent=self.write_n)
            self.sim.u0[k].upload(self.sim.stream, self.in_s[k,:,:], extent=self.write_s)
        
        #Wait for sending to complete
        for comm in comm_send:
            comm.wait()
        
        
        
        ####
        # Then transfer east-west including ghost cells that have been filled in by north-south transfer above
        ####
        
        #Download from the GPU
        for k in range(self.nvars):
            self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_e[k,:,:], async=True, extent=self.read_e)
            self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_w[k,:,:], async=True, extent=self.read_w)
        self.sim.stream.synchronize()
        
        #Send to east/west neighbours
        comm_send = []
        comm_send += [self.grid.comm.Isend(self.out_e, dest=self.east, tag=4*self.nt + 2)]
        comm_send += [self.grid.comm.Isend(self.out_w, dest=self.west, tag=4*self.nt + 3)]
        
        #Receive from east/west neighbors
        comm_recv = []
        comm_recv += [self.grid.comm.Irecv(self.in_w, source=self.west, tag=4*self.nt + 2)]
        comm_recv += [self.grid.comm.Irecv(self.in_e, source=self.east, tag=4*self.nt + 3)]
        
        #Wait for incoming transfers to complete
        for comm in comm_recv:
            comm.wait()
        
        #Upload to the GPU
        for k in range(self.nvars):
            self.sim.u0[k].upload(self.sim.stream, self.in_e[k,:,:], extent=self.write_e)
            self.sim.u0[k].upload(self.sim.stream, self.in_w[k,:,:], extent=self.write_w)
        
        #Wait for sending to complete
        for comm in comm_send:
            comm.wait()
    
    
    
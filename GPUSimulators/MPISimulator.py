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

class MPISimulator(Simulator.BaseSimulator):
    def __init__(self, sim, comm):
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
        self.comm = comm
        self.rank = comm.rank
        
        #Get global dimensions
        self.grid = MPISimulator.getFactors(self.comm.size, 2)
        
        #Get neighbor node ids
        self.east = self.getEast()
        self.west = self.getWest()
        self.north = self.getNorth()
        self.south = self.getSouth()
        
        #Get local dimensions
        self.gc_x = int(self.sim.u0[0].x_halo)
        self.gc_y = int(self.sim.u0[0].y_halo)
        self.nx = int(self.sim.nx)
        self.ny = int(self.sim.ny)
        self.nvars = 3
        
        #Allocate data for receiving
        #Note that east and west also transfer ghost cells
        #whilst north/south only transfer internal cells
        self.in_e = np.empty((self.nvars, self.ny + 2*self.gc_y, self.gc_x), dtype=np.float32)
        self.in_w = np.empty((self.nvars, self.ny + 2*self.gc_y, self.gc_x), dtype=np.float32)
        self.in_n = np.empty((self.nvars,             self.gc_y,   self.nx), dtype=np.float32)
        self.in_s = np.empty((self.nvars,             self.gc_y,   self.nx), dtype=np.float32)
        
        #Allocate data for sending
        self.out_e = np.empty((self.nvars, self.ny + 2*self.gc_y, self.gc_x), dtype=np.float32)
        self.out_w = np.empty((self.nvars, self.ny + 2*self.gc_y, self.gc_x), dtype=np.float32)
        self.out_n = np.empty((self.nvars,             self.gc_y,   self.nx), dtype=np.float32)
        self.out_s = np.empty((self.nvars,             self.gc_y,   self.nx), dtype=np.float32)
        
        #Set regions for ghost cells to read from
        self.read_e = [  self.nx,         0, self.gc_x, self.ny + 2*self.gc_y]
        self.read_w = [self.gc_x,         0, self.gc_x, self.ny + 2*self.gc_y]
        self.read_n = [self.gc_x,   self.ny,   self.nx,             self.gc_y]
        self.read_s = [self.gc_x, self.gc_y,   self.nx,             self.gc_y]
        
        #Set regions for ghost cells to write to
        self.write_e = [self.nx+self.gc_x,                 0, self.gc_x, self.ny + 2*self.gc_y]
        self.write_w = [                0,                 0, self.gc_x, self.ny + 2*self.gc_y]
        self.write_n = [        self.gc_x, self.ny+self.gc_y,   self.nx,             self.gc_y]
        self.write_s = [        self.gc_x,                 0,   self.nx,             self.gc_y]
        
        #Initialize ghost cells
        self.exchange()
        
        self.logger.debug("Simlator rank {:d} created ".format(self.rank))
    
        
    def substep(self, dt, step_number):
        self.sim.substep(dt, step_number)
        self.exchange()
    
    def download(self):
        raise(NotImplementedError("Needs to be implemented!"))
    
    def synchronize(self):
        raise(NotImplementedError("Needs to be implemented!"))
    
    def check(self):
        return self.sim.check()
        
    def computeDt(self):
        raise(NotImplementedError("Needs to be implemented!"))
        
    def exchange(self):
        #Shorthands for dimensions
        gc_x = self.gc_x
        gc_y = self.gc_y
        nx = self.nx
        ny = self.ny
        
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
        comm_send += [self.comm.Isend(self.out_n, dest=self.north, tag=0)]
        comm_send += [self.comm.Isend(self.out_s, dest=self.south, tag=1)]
        
        #Receive from north/south neighbors
        comm_recv = []
        comm_recv += [self.comm.Irecv(self.in_n, source=self.north, tag=1)]
        comm_recv += [self.comm.Irecv(self.in_s, source=self.south, tag=0)]
        
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
        # Fixme: This can be optimized by overlapping the GPU transfer with the pervious MPI transfer if the corners
        # har handled on the CPU
        ####
        
        #Download from the GPU
        for k in range(self.nvars):
            self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_e[k,:,:], async=True, extent=self.read_e)
            self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_w[k,:,:], async=True, extent=self.read_w)
        self.sim.stream.synchronize()
        
        #Send to east/west neighbours
        comm_send = []
        comm_send += [self.comm.Isend(self.out_e, dest=self.east, tag=2)]
        comm_send += [self.comm.Isend(self.out_w, dest=self.west, tag=3)]
        
        #Receive from east/west neighbors
        comm_recv = []
        comm_recv += [self.comm.Irecv(self.in_e, source=self.east, tag=3)]
        comm_recv += [self.comm.Irecv(self.in_w, source=self.west, tag=2)]
        
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
    
    
    def getCoordinate(self, rank):
        i = (rank  % self.grid[0])
        j = (rank // self.grid[0])
        return i, j

    def getRank(self, i, j):
        return j*self.grid[0] + i

    def getEast(self):
        i, j = self.getCoordinate(self.rank)
        i = (i+1) % self.grid[0]
        return self.getRank(i, j)

    def getWest(self):
        i, j = self.getCoordinate(self.rank)
        i = (i+self.grid[0]-1) % self.grid[0]
        return self.getRank(i, j)

    def getNorth(self):
        i, j = self.getCoordinate(self.rank)
        j = (j+1) % self.grid[1]
        return self.getRank(i, j)

    def getSouth(self):
        i, j = self.getCoordinate(self.rank)
        j = (j+self.grid[1]-1) % self.grid[1]
        return self.getRank(i, j)
    
    def getFactors(number, num_factors):
        # Adapted from https://stackoverflow.com/questions/28057307/factoring-a-number-into-roughly-equal-factors
        # Original code by https://stackoverflow.com/users/3928385/ishamael

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

        assert(isinstance(number, int))
        assert(isinstance(num_factors, int))

        factors = dp(number, num_factors)[1]

        if (len(factors) < num_factors):
            #Split problematic 4
            if (4 in factors):
                factors.remove(4)
                factors.append(2)
                factors.append(2)

        #Pad with ones to guarantee num_factors
        factors = factors + [1]*(num_factors - len(factors))
        return factors
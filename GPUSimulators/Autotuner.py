# -*- coding: utf-8 -*-

"""
This python module implements the different helper functions and 
classes

Copyright (C) 2018  SINTEF ICT

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

import os
import gc
import numpy as np
import logging
from socket import gethostname

import pycuda.driver as cuda


from GPUSimulators import Common, Simulator

class Autotuner:
    def __init__(self, 
                nx=2048, ny=2048, 
                block_widths=range(8, 32, 2),
                block_heights=range(8, 32, 2)):
        logger = logging.getLogger(__name__)
        self.filename = "autotuning_data_" + gethostname() + ".npz"
        self.nx = nx
        self.ny = ny
        self.block_widths = block_widths
        self.block_heights = block_heights
        self.performance = {}


    def benchmark(self, simulator, force=False):
        logger = logging.getLogger(__name__)
        
        #Run through simulators and benchmark
        key = str(simulator.__name__)
        logger.info("Benchmarking %s to %s", key, self.filename)
        
        #If this simulator has been benchmarked already, skip it
        if (force==False and os.path.isfile(self.filename)):
            with np.load(self.filename) as data:
                if key in data["simulators"]:
                    logger.info("%s already benchmarked - skipping", key)
                    return
    
        # Set arguments to send to the simulators during construction
        context = Common.CudaContext(autotuning=False)
        g = 9.81
        h0, hu0, hv0, dx, dy, dt = Autotuner.gen_test_data(nx=self.nx, ny=self.ny, g=g)
        arguments = {
            'context': context,
            'h0': h0, 'hu0': hu0, 'hv0': hv0,
            'nx': self.nx, 'ny': self.ny,
            'dx': dx, 'dy': dy, 'dt': 0.9*dt,
            'g': g
        } 
             
        # Load existing data into memory
        benchmark_data = {
                "simulators": [],
        }
        if (os.path.isfile(self.filename)):
            with np.load(self.filename) as data:
                for k, v in data.items():
                    benchmark_data[k] = v
   
        # Run benchmark
        benchmark_data[key + "_megacells"] = Autotuner.benchmark_single_simulator(simulator, arguments, self.block_widths, self.block_heights)
        benchmark_data[key + "_block_widths"] = self.block_widths
        benchmark_data[key + "_block_heights"] = self.block_heights
        benchmark_data[key + "_arguments"] = str(arguments)
        
        existing_sims = benchmark_data["simulators"]
        if (isinstance(existing_sims, np.ndarray)):
            existing_sims = existing_sims.tolist()
        if (key not in existing_sims):
            benchmark_data["simulators"] = existing_sims + [key]

        # Save to file
        np.savez_compressed(self.filename, **benchmark_data)


            
    """
    Function which reads a numpy file with autotuning data
    and reports the maximum performance and block size
    """
    def get_peak_performance(self, simulator):
        logger = logging.getLogger(__name__)
        
        assert issubclass(simulator, Simulator.BaseSimulator)
        key = simulator.__name__
        
        if (key in self.performance):
            return self.performance[key]
        else:
            #Run simulation if required
            if (not os.path.isfile(self.filename)):
                logger.debug("Could not get autotuned peak performance for %s: benchmarking", key)
                self.benchmark(simulator)
            
            with np.load(self.filename) as data:
                if key not in data['simulators']:
                    logger.debug("Could not get autotuned peak performance for %s: benchmarking", key)
                    data.close()
                    self.benchmark(simulator)
                    data = np.load(self.filename)
                
                def find_max_index(megacells):
                    max_index = np.nanargmax(megacells)
                    return np.unravel_index(max_index, megacells.shape)
                
                megacells = data[key + '_megacells']
                block_widths = data[key + '_block_widths']
                block_heights = data[key + '_block_heights']
                j, i = find_max_index(megacells)
                
                self.performance[key] = { "block_width": block_widths[i],
                                         "block_height": block_heights[j],
                                         "megacells": megacells[j, i] }
                logger.debug("Returning %s as peak performance parameters", self.performance[key])
                return self.performance[key]
        
            #This should never happen
            raise "Something wrong: Could not get autotuning data!"
            return None
        
        
                
    """
    Runs a set of benchmarks for a single simulator
    """
    def benchmark_single_simulator(simulator, arguments, block_widths, block_heights):
        logger = logging.getLogger(__name__)
        
        megacells = np.empty((len(block_heights), len(block_widths)))
        megacells.fill(np.nan)

        logger.debug("Running %d benchmarks with %s", len(block_heights)*len(block_widths), simulator.__name__)
        
        sim_arguments = arguments.copy()
                    
        with Common.Timer(simulator.__name__) as t:
            for j, block_height in enumerate(block_heights):
                sim_arguments.update({'block_height': block_height})
                for i, block_width in enumerate(block_widths):
                    sim_arguments.update({'block_width': block_width})
                    megacells[j, i] = Autotuner.run_benchmark(simulator, sim_arguments)
                        

        logger.debug("Completed %s in %f seconds", simulator.__name__, t.secs)

        return megacells
            
            
    """
    Runs a benchmark, and returns the number of megacells achieved
    """
    def run_benchmark(simulator, arguments, timesteps=10, warmup_timesteps=2):
        logger = logging.getLogger(__name__)
        
        #Initialize simulator
        try:
            sim = simulator(**arguments)
        except:
            #An exception raised - not possible to continue
            logger.debug("Failed creating %s with arguments %s", simulator.__name__, str(arguments))
            return np.nan
        
        #Create timer events
        start = cuda.Event()
        end = cuda.Event()
        
        #Warmup
        for i in range(warmup_timesteps):
            sim.stepEuler(sim.dt)
            
        #Run simulation with timer        
        start.record(sim.stream)
        for i in range(timesteps):
            sim.stepEuler(sim.dt)
        end.record(sim.stream)
        
        #Synchronize end event
        end.synchronize()
        
        #Compute megacells
        gpu_elapsed = end.time_since(start)*1.0e-3
        megacells = (sim.nx*sim.ny*timesteps / (1000*1000)) / gpu_elapsed

        #Sanity check solution
        h, hu, hv = sim.download()
        sane = True
        sane = sane and Autotuner.sanity_check(h, 0.3, 0.7)
        sane = sane and Autotuner.sanity_check(hu, -0.2, 0.2)
        sane = sane and Autotuner.sanity_check(hv, -0.2, 0.2)
        
        if (sane):
            logger.debug("%s [%d x %d] succeeded: %f megacells, gpu elapsed %f", simulator.__name__, arguments["block_width"], arguments["block_height"], megacells, gpu_elapsed)
            return megacells
        else:
            logger.debug("%s [%d x %d] failed: gpu elapsed %f", simulator.__name__, arguments["block_width"], arguments["block_height"], gpu_elapsed)
            return np.nan
        
        
        
    """
    Generates test dataset
    """
    def gen_test_data(nx, ny, g):
        width = 100.0
        height = 100.0
        dx = width / float(nx)
        dy = height / float(ny)

        x_center = dx*nx/2.0
        y_center = dy*ny/2.0

        #Create a gaussian "dam break" that will not form shocks
        size = width / 5.0
        dt = 10**10
        
        h  = np.zeros((ny, nx), dtype=np.float32); 
        hu = np.zeros((ny, nx), dtype=np.float32);
        hv = np.zeros((ny, nx), dtype=np.float32);
        
        x = dx*(np.arange(0, nx, dtype=np.float32)+0.5) - x_center
        y = dy*(np.arange(0, ny, dtype=np.float32)+0.5) - y_center
        xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
        r = np.sqrt(xv**2 + yv**2)
        xv = None
        yv = None
        gc.collect()
            
        #Generate highres
        h = 0.5 + 0.1*0.5*(1.0 + np.cos(np.pi*r/size)) * (r < size)
        hu = 0.1*0.5*(1.0 + np.cos(np.pi*r/size)) * (r < size)
        hv = 0.1*0.5*(1.0 + np.cos(np.pi*r/size)) * (r < size)
        
        scale = 0.7
        max_h_estimate = 0.6
        max_u_estimate = 0.1*np.sqrt(2.0)
        dx = width/nx
        dy = height/ny
        dt = scale * min(dx, dy) / (max_u_estimate + np.sqrt(g*max_h_estimate))
        
        return h, hu, hv, dx, dy, dt
        
    """
    Checks that a variable is "sane"
    """
    def sanity_check(variable, bound_min, bound_max):
        maxval = np.amax(variable)
        minval = np.amin(variable)
        if (np.isnan(maxval) 
                or np.isnan(minval)
                or maxval > bound_max
                or minval < bound_min):
            return False
        else:
            return True
# -*- coding: utf-8 -*-

"""
This python module implements Cuda context handling

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


from GPUSimulators.Simulator import BoundaryCondition
import numpy as np

def genShockBubble(nx, ny, gamma):
    """
    Generate Shock-bubble interaction case for the Euler equations
    """
    
    width = 4.0
    height = 1.0
    dx = width / float(nx)
    dy = height / float(ny)
    g = 0.0


    rho = np.ones((ny, nx), dtype=np.float32)
    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    E = np.zeros((ny, nx), dtype=np.float32)
    p = np.ones((ny, nx), dtype=np.float32)

    x_center = 0.5
    y_center = 0.5
    x = np.linspace(0.5*dx, nx*dx-0.5*dx, nx, dtype=np.float32) - x_center
    y = np.linspace(0.5*dy, ny*dy-0.5*dy, ny, dtype=np.float32) - y_center
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
       
    #Bubble
    radius = 0.25
    bubble = np.sqrt(xv**2+yv**2) <= radius
    rho = np.where(bubble, 0.1, rho)
    
    #Left boundary
    left = (xv - xv.min() < 0.1)
    rho = np.where(left, 3.81250, rho)
    u = np.where(left, 2.57669, u)
    
    #Energy
    p = np.where(left, 10.0, p)
    E = 0.5*rho*(u**2+v**2) + p/(gamma-1.0)
    
    #Estimate dt
    scale = 0.45
    max_rho_estimate = 5.0
    max_u_estimate = 5.0
    dx = width/nx
    dy = height/ny
    dt = scale * min(dx, dy) / (max_u_estimate + np.sqrt(gamma*max_rho_estimate))

    bc = BoundaryCondition({
        'north': BoundaryCondition.Type.Reflective,
        'south': BoundaryCondition.Type.Reflective,
        'east': BoundaryCondition.Type.Periodic,
        'west': BoundaryCondition.Type.Periodic
    })
    
    #Construct simulator
    arguments = {
        'rho': rho, 'rho_u': rho*u, 'rho_v': rho*v, 'E': E,
        'nx': nx, 'ny': ny,
        'dx': dx, 'dy': dy, 'dt': dt,
        'g': g,
        'gamma': gamma,
        'boundary_conditions': bc
    } 
    return arguments

    
    
    
    
    
    
def genKelvinHelmholtz(nx, ny, gamma, roughness=0.125):
    """
    Roughness parameter in (0, 1.0] determines how "squiggly" 
    the interface betweeen the zones is
    """
    
    def genZones(nx, ny, n):
        """
        Generates the zones of the two fluids of K-H
        """
        zone = np.zeros((ny, nx), dtype=np.int32)

        dx = 1.0 / nx
        dy = 1.0 / ny

        def genSmoothRandom(nx, n):
            assert (n <= nx), "Number of generated points nx must be larger than n"
            
            if n == nx:
                return np.random.random(nx)-0.5
            else:
                from scipy.interpolate import interp1d

                #Control points and interpolator
                xp = np.linspace(0.0, 1.0, n)
                yp = np.random.random(n) - 0.5
                f = interp1d(xp, yp, kind='cubic')

                #Interpolation points
                x = np.linspace(0.0, 1.0, nx)
                return f(x)



        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)

        _, y = np.meshgrid(x, y)

        #print(y+a[0])

        a = genSmoothRandom(nx, n)*dy
        zone = np.where(y > 0.25+a, zone, 1)

        a = genSmoothRandom(nx, n)*dy
        zone = np.where(y < 0.75+a, zone, 1)
        
        return zone
        
    width = 2.0
    height = 1.0
    dx = width / float(nx)
    dy = height / float(ny)
    g = 0.0
    gamma = 1.4

    rho = np.empty((ny, nx), dtype=np.float32)
    u = np.empty((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    p = 2.5*np.ones((ny, nx), dtype=np.float32)

    #Generate the different zones    
    zones = genZones(nx, ny, max(1, min(nx, int(nx*roughness))))
    
    #Zone 0
    zone0 = zones == 0
    rho = np.where(zone0, 1.0, rho)
    u = np.where(zone0, 0.5, u)
    
    #Zone 1
    zone1 = zones == 1
    rho = np.where(zone1, 2.0, rho)
    u = np.where(zone1, -0.5, u)
    
    E = 0.5*rho*(u**2+v**2) + p/(gamma-1.0)
    
    #Estimate dt
    scale = 0.9
    max_rho_estimate = 3.0
    max_u_estimate = 2.0
    dx = width/nx
    dy = height/ny
    dt = scale * min(dx, dy) / (max_u_estimate + np.sqrt(gamma*max_rho_estimate))
    
    
    bc = BoundaryCondition({
        'north': BoundaryCondition.Type.Periodic,
        'south': BoundaryCondition.Type.Periodic,
        'east': BoundaryCondition.Type.Periodic,
        'west': BoundaryCondition.Type.Periodic
    })
    
    #Construct simulator
    arguments = {
        'rho': rho, 'rho_u': rho*u, 'rho_v': rho*v, 'E': E,
        'nx': nx, 'ny': ny,
        'dx': dx, 'dy': dy, 'dt': dt,
        'g': g,
        'gamma': gamma,
        'boundary_conditions': bc
    } 
    
    return arguments
    
    
    
def genRayleighTaylor(nx, ny, gamma, version=0):
    """
    Generates Rayleigh-Taylor instability case
    """
    width = 0.5
    height = 1.5
    dx = width / float(nx)
    dy = height / float(ny)
    g = 0.1

    rho = np.zeros((ny, nx), dtype=np.float32)
    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    p = np.zeros((ny, nx), dtype=np.float32)
    
    x = np.linspace(0.5*dx, nx*dx-0.5*dx, nx, dtype=np.float32)-width*0.5
    y = np.linspace(0.5*dy, ny*dy-0.5*dy, ny, dtype=np.float32)-height*0.5
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    
    #This gives a squigly interfact
    if (version == 0):
        y_threshold = 0.01*np.cos(2*np.pi*np.abs(x)/0.5)
        rho = np.where(yv <= y_threshold, 1.0, rho)
        rho = np.where(yv > y_threshold, 2.0, rho)
    elif (version == 1):
        rho = np.where(yv <= 0.0, 1.0, rho)
        rho = np.where(yv > 0.0, 2.0, rho)
        v = 0.01*(1.0 + np.cos(2*np.pi*xv/0.5))/4
    else:
        assert False, "Invalid version"
    
    p = 2.5 - rho*g*yv
    E = 0.5*rho*(u**2+v**2) + p/(gamma-1.0)
    
    #Estimate dt
    scale = 0.9
    max_rho_estimate = 3.0
    max_u_estimate = 1.0
    dx = width/nx
    dy = height/ny
    dt = scale * min(dx, dy) / (max_u_estimate + np.sqrt(gamma*max_rho_estimate))
    
    bc = BoundaryCondition({
        'north': BoundaryCondition.Type.Reflective,
        'south': BoundaryCondition.Type.Reflective,
        'east': BoundaryCondition.Type.Reflective,
        'west': BoundaryCondition.Type.Reflective
    })
    
    #Construct simulator
    arguments = {
        'rho': rho, 'rho_u': rho*u, 'rho_v': rho*v, 'E': E,
        'nx': nx, 'ny': ny,
        'dx': dx, 'dy': dy, 'dt': dt,
        'g': g,
        'gamma': gamma,
        'boundary_conditions': bc
    } 

    return arguments
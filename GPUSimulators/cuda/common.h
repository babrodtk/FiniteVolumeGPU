/*
This OpenCL kernel implements the Kurganov-Petrova numerical scheme 
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
*/

#pragma once


/**
  * Float3 operators 
  */
inline __device__ float3 operator*(const float a, const float3 b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __device__ float3 operator/(const float3 a, const float b) {
    return make_float3(a.x/b, a.y/b, a.z/b);
}

inline __device__ float3 operator-(const float3 a, const float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ float3 operator+(const float3 a, const float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

/**
  * Float4 operators 
  */
inline __device__ float4 operator*(const float a, const float4 b) {
    return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}

inline __device__ float4 operator/(const float4 a, const float b) {
    return make_float4(a.x/b, a.y/b, a.z/b, a.w/b);
}

inline __device__ float4 operator-(const float4 a, const float4 b) {
    return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

inline __device__ float4 operator+(const float4 a, const float4 b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}




inline __device__ __host__ float clamp(const float f, const float a, const float b) {
    return fmaxf(a, fminf(f, b));
}

inline __device__ __host__ int clamp(const int f, const int a, const int b) {
    return (f < b) ? ( (f > a) ? f : a) : b;
}





__device__ float desingularize(float x_, float eps_) {
    return copysign(1.0f, x_)*fmaxf(fabsf(x_), fminf(x_*x_/(2.0f*eps_)+0.5f*eps_, eps_));
}








/**
  * Returns the step stored in the leftmost 16 bits 
  * of the 32 bit step-order integer
  */
inline __device__ int getStep(int step_order_) {
    return step_order_ >> 16;
}

/**
  * Returns the order stored in the rightmost 16 bits 
  * of the 32 bit step-order integer
  */
inline __device__ int getOrder(int step_order_) {
    return step_order_ & 0x0000FFFF;
}


enum BoundaryCondition {
    Dirichlet = 0,
    Neumann = 1,
    Periodic = 2,
    Reflective = 3
};

inline __device__ BoundaryCondition getBCNorth(int bc_) {
    return static_cast<BoundaryCondition>(bc_ & 0x0000000F);
}

inline __device__ BoundaryCondition getBCSouth(int bc_) {
    return static_cast<BoundaryCondition>((bc_ >> 8) & 0x0000000F);
}

inline __device__ BoundaryCondition getBCEast(int bc_) {
    return static_cast<BoundaryCondition>((bc_ >> 16) & 0x0000000F);
}

inline __device__ BoundaryCondition getBCWest(int bc_) {
    return static_cast<BoundaryCondition>(bc_ >> 24);
}






/**
  * Alter the index l so that it gives periodic boundary conditions when reading
  */
template<int ghost_cells>
inline __device__ int handlePeriodicBoundaryX(int k, int nx_, int boundary_conditions_) {
    const int gc_pad = 2*ghost_cells;
    
    //West boundary: add an offset to read from east of domain
    if ((k < gc_pad) 
            && getBCWest(boundary_conditions_) == Periodic) {
        k += (nx_+2*ghost_cells - 2*gc_pad);
    }
    //East boundary: subtract an offset to read from west of domain
    else if ((k >= nx_+2*ghost_cells-gc_pad) 
            && getBCEast(boundary_conditions_) == Periodic) {
        k -= (nx_+2*ghost_cells - 2*gc_pad);
    }
    
    return k;
}

/**
  * Alter the index l so that it gives periodic boundary conditions when reading
  */
template<int ghost_cells>
inline __device__ int handlePeriodicBoundaryY(int l, int ny_, int boundary_conditions_) {
    const int gc_pad = 2*ghost_cells;
    
    //South boundary: add an offset to read from north of domain
    if ((l < gc_pad) 
            && getBCSouth(boundary_conditions_) == Periodic) {
        l += (ny_+2*ghost_cells - 2*gc_pad);
    }
    //North boundary: subtract an offset to read from south of domain
    else if ((l >= ny_+2*ghost_cells-gc_pad) 
            && getBCNorth(boundary_conditions_) == Periodic) {
        l -= (ny_+2*ghost_cells - 2*gc_pad);
    }
    
    return l;
}


template<int block_width, int block_height, int ghost_cells, int sign_x, int sign_y>
inline __device__ int handleReflectiveBoundary(
                float Q[block_height+2*ghost_cells][block_width+2*ghost_cells], 
                const int nx_, const int ny_,
                const int boundary_conditions_) {
    //Handle reflective boundary conditions
    if (getBCNorth(boundary_conditions_) == Reflective) {
        bcNorthReflective<block_width, block_height, ghost_cells, sign_y>(Q, nx_, ny_);
        __syncthreads();
    }
    if (getBCSouth(boundary_conditions_) == Reflective) {
        bcSouthReflective<block_width, block_height, ghost_cells, sign_y>(Q, nx_, ny_);
        __syncthreads();
    }
    if (getBCEast(boundary_conditions_) == Reflective) {
        bcEastReflective<block_width, block_height, ghost_cells, sign_x>(Q, nx_, ny_);
        __syncthreads();
    }
    if (getBCWest(boundary_conditions_) == Reflective) {
        bcWestReflective<block_width, block_height, ghost_cells, sign_x>(Q, nx_, ny_);
        __syncthreads();
    }
}

/**
  * Reads a block of data with ghost cells
  */
template<int block_width, int block_height, int ghost_cells, int sign_x, int sign_y>
inline __device__ void readBlock(float* ptr_, int pitch_,
                float Q[block_height+2*ghost_cells][block_width+2*ghost_cells], 
                const int nx_, const int ny_,
                const int boundary_conditions_) {
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Read into shared memory
    //Loop over all variables
    for (int j=threadIdx.y; j<block_height+2*ghost_cells; j+=block_height) {
        //Handle periodic boundary conditions here
        int l = handlePeriodicBoundaryY<ghost_cells>(by + j, ny_, boundary_conditions_);
        l = min(l, ny_+2*ghost_cells-1);
        float* row = (float*) ((char*) ptr_ + pitch_*l);
        
        for (int i=threadIdx.x; i<block_width+2*ghost_cells; i+=block_width) {
            //Handle periodic boundary conditions here
            int k = handlePeriodicBoundaryX<ghost_cells>(bx + i, nx_, boundary_conditions_);
            k = min(k, nx_+2*ghost_cells-1);
            
            //Read from global memory
            Q[j][i] = row[k];
        }
    }
    __syncthreads();
    
    handleReflectiveBoundary<block_width, block_height, ghost_cells, sign_x, sign_y>(Q, nx_, ny_, boundary_conditions_);
}




/**
  * Writes a block of data to global memory for the shallow water equations.
  */
template<int block_width, int block_height, int ghost_cells>
inline __device__ void writeBlock(float* ptr_, int pitch_,
                 float shmem[block_height+2*ghost_cells][block_width+2*ghost_cells],
                 const int width, const int height,
                 int rk_step_, int rk_order_) {
    
    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + ghost_cells;
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + ghost_cells;
    
    //Only write internal cells
    if (ti < width+ghost_cells && tj < height+ghost_cells) {
        //Index of thread within block
        const int tx = threadIdx.x + ghost_cells;
        const int ty = threadIdx.y + ghost_cells;
        
        float* const row  = (float*) ((char*) ptr_ + pitch_*tj);
        
        //Handle runge-kutta timestepping here
        row[ti] = shmem[ty][tx];
        
        /**
          * SSPRK2
          * u^1   = u^n + dt*f(u^n)
          * u^n+1 = 1/2*u^n + 1/2*(u^1 + dt*f(u^1))
          *
          * SSPRK3
          * u^1   = u^n + dt*f(u^n)
          * u^2   = 3/4 * u^n + 1/4 * (u^1 + dt*f(u^1))
          * u^n+1 = 1/3 * u^n + 2/3 * (u^2 + dt*f(u^2))
          */
        
        /*
        if (rk_order_ == 2 && rk_step_ == 1) {
            row[ti] = 0.5f*(row[ti] + shmem[ty][tx]);
        }
        else {
            row[ti] = shmem[ty][tx];
        }*/
    }
}














// West boundary
template<int block_width, int block_height, int ghost_cells, int sign>
__device__ void bcWestReflective(float Q[block_height+2*ghost_cells][block_width+2*ghost_cells], const int nx_, const int ny_) {
    for (int j=threadIdx.y; j<block_height+2*ghost_cells; j+= block_height) {
        const int i = threadIdx.x + ghost_cells;
        const int ti = blockDim.x*blockIdx.x + i;
        
        if (ti == ghost_cells) {
            Q[j][i-1] = sign*Q[j][i];
        }
        if (ghost_cells >= 2 && ti == ghost_cells + 1) {
            Q[j][i-3] = sign*Q[j][i];
        }
        if (ghost_cells >= 3 && ti == ghost_cells + 2) {
            Q[j][i-5] = sign*Q[j][i];
        }
        if (ghost_cells >= 4 && ti == ghost_cells + 3) {
            Q[j][i-7] = sign*Q[j][i];
        }
        if (ghost_cells >= 5 && ti == ghost_cells + 4) {
            Q[j][i-9] = sign*Q[j][i];
        }
    }
}


// East boundary
template<int block_width, int block_height, int ghost_cells, int sign>
__device__ void bcEastReflective(float Q[block_height+2*ghost_cells][block_width+2*ghost_cells], const int nx_, const int ny_) {
    for (int j=threadIdx.y; j<block_height+2*ghost_cells; j+= block_height) {
        const int i = threadIdx.x + ghost_cells;
        const int ti = blockDim.x*blockIdx.x + i;
        
        if (ti == nx_ + ghost_cells - 1) {
            Q[j][i+1] = sign*Q[j][i];
        }
        if (ghost_cells >= 2 && ti == nx_ + ghost_cells - 2) {
            Q[j][i+3] = sign*Q[j][i];
        }
        if (ghost_cells >= 3 && ti == nx_ + ghost_cells - 3) {
            Q[j][i+5] = sign*Q[j][i];
        }
        if (ghost_cells >= 4 && ti == nx_ + ghost_cells - 4) {
            Q[j][i+7] = sign*Q[j][i];
        }
        if (ghost_cells >= 5 && ti == nx_ + ghost_cells - 5) {
            Q[j][i+9] = sign*Q[j][i];
        }
    }
}
    
    
// South boundary
template<int block_width, int block_height, int ghost_cells, int sign>
__device__ void bcSouthReflective(float Q[block_height+2*ghost_cells][block_width+2*ghost_cells], const int nx_, const int ny_) {
    for (int i=threadIdx.x; i<block_width+2*ghost_cells; i+= block_width) {
        const int j = threadIdx.y + ghost_cells;
        const int tj = blockDim.y*blockIdx.y + j;

        if (tj == ghost_cells) {
            Q[j-1][i] = sign*Q[j][i];
        }
        if (ghost_cells >= 2 && tj == ghost_cells + 1) {
            Q[j-3][i] = sign*Q[j][i];
        }
        if (ghost_cells >= 3 && tj == ghost_cells + 2) {
            Q[j-5][i] = sign*Q[j][i];
        }
        if (ghost_cells >= 4 && tj == ghost_cells + 3) {
            Q[j-7][i] = sign*Q[j][i];
        }
        if (ghost_cells >= 5 && tj == ghost_cells + 4) {
            Q[j-9][i] = sign*Q[j][i];
        }
    }
}
        
        
        
    
// North boundary
template<int block_width, int block_height, int ghost_cells, int sign>
__device__ void bcNorthReflective(float Q[block_height+2*ghost_cells][block_width+2*ghost_cells], const int nx_, const int ny_) {
    for (int i=threadIdx.x; i<block_width+2*ghost_cells; i+= block_width) {
        const int j = threadIdx.y + ghost_cells;
        const int tj = blockDim.y*blockIdx.y + j;
        
        if (tj == ny_ + ghost_cells - 1) {
            Q[j+1][i] = sign*Q[j][i];
        }
        if (ghost_cells >= 2 && tj == ny_ + ghost_cells - 2) {
            Q[j+3][i] = sign*Q[j][i];
        }
        if (ghost_cells >= 3 && tj == ny_ + ghost_cells - 3) {
            Q[j+5][i] = sign*Q[j][i];
        }
        if (ghost_cells >= 4 && tj == ny_ + ghost_cells - 4) {
            Q[j+7][i] = sign*Q[j][i];
        }
        if (ghost_cells >= 5 && tj == ny_ + ghost_cells - 5) {
            Q[j+9][i] = sign*Q[j][i];
        }
    }
}






















template<int block_width, int block_height, int ghost_cells, int vars>
__device__ void evolveF(float Q[vars][block_height+2*ghost_cells][block_width+2*ghost_cells],
              float F[vars][block_height+2*ghost_cells][block_width+2*ghost_cells],
              const float dx_, const float dt_) {
    for (int var=0; var < vars; ++var) {
        for (int j=threadIdx.y; j<block_height+2*ghost_cells; j+=block_height) {
            for (int i=threadIdx.x+ghost_cells; i<block_width+ghost_cells; i+=block_width) {
                Q[var][j][i] = Q[var][j][i] + (F[var][j][i-1] - F[var][j][i]) * dt_ / dx_;
            }
        }
    }
}






/**
  * Evolves the solution in time along the y axis (dimensional splitting)
  */
template<int block_width, int block_height, int ghost_cells, int vars>
__device__ void evolveG(float Q[vars][block_height+2*ghost_cells][block_width+2*ghost_cells],
              float G[vars][block_height+2*ghost_cells][block_width+2*ghost_cells],
              const float dy_, const float dt_) {
    for (int var=0; var < vars; ++var) {
        for (int j=threadIdx.y+ghost_cells; j<block_height+ghost_cells; j+=block_height) {
            for (int i=threadIdx.x; i<block_width+2*ghost_cells; i+=block_width) {
                Q[var][j][i] = Q[var][j][i] + (G[var][j-1][i] - G[var][j][i]) * dt_ / dy_;
            }
        }
    }
}





/**
  * Helper function for debugging etc.
  */
template<int shmem_width, int shmem_height, int vars>
__device__ void memset(float Q[vars][shmem_height][shmem_width], float value) {
    for (int k=0; k<vars; ++k) {
        for (int j=threadIdx.y; j<shmem_height; ++j) {
            for (int i=threadIdx.x; i<shmem_width; ++i) {
                Q[k][j][i] = value;
            }
        }
    }
} 
















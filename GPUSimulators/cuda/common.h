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
    return static_cast<BoundaryCondition>((bc_ >> 24) & 0x0000000F);
}

inline __device__ BoundaryCondition getBCSouth(int bc_) {
    return static_cast<BoundaryCondition>((bc_ >> 16) & 0x0000000F);
}

inline __device__ BoundaryCondition getBCEast(int bc_) {
    return static_cast<BoundaryCondition>((bc_ >> 8) & 0x0000000F);
}

inline __device__ BoundaryCondition getBCWest(int bc_) {
    return static_cast<BoundaryCondition>((bc_ >> 0) & 0x0000000F);
}


// West boundary
template<int w, int h, int gc_x, int gc_y, int sign>
__device__ void bcWestReflective(float Q[h+2*gc_y][w+2*gc_x], 
                                const int nx_, const int ny_) {
    for (int j=threadIdx.y; j<h+2*gc_y; j+=h) {
        const int i = threadIdx.x + gc_x;
        const int ti = blockDim.x*blockIdx.x + i;
        
        if (gc_x >= 1 && ti == gc_x) {
            Q[j][i-1] = sign*Q[j][i];
        }
        if (gc_x >= 2 && ti == gc_x + 1) {
            Q[j][i-3] = sign*Q[j][i];
        }
        if (gc_x >= 3 && ti == gc_x + 2) {
            Q[j][i-5] = sign*Q[j][i];
        }
        if (gc_x >= 4 && ti == gc_x + 3) {
            Q[j][i-7] = sign*Q[j][i];
        }
        if (gc_x >= 5 && ti == gc_x + 4) {
            Q[j][i-9] = sign*Q[j][i];
        }
    }
}


// East boundary
template<int w, int h, int gc_x, int gc_y, int sign>
__device__ void bcEastReflective(float Q[h+2*gc_y][w+2*gc_x], 
                                const int nx_, const int ny_) {
    for (int j=threadIdx.y; j<h+2*gc_y; j+=h) {
        const int i = threadIdx.x + gc_x;
        const int ti = blockDim.x*blockIdx.x + i;
        
        if (gc_x >= 1 && ti == nx_ + gc_x - 1) {
            Q[j][i+1] = sign*Q[j][i];
        }
        if (gc_x >= 2 && ti == nx_ + gc_x - 2) {
            Q[j][i+3] = sign*Q[j][i];
        }
        if (gc_x >= 3 && ti == nx_ + gc_x - 3) {
            Q[j][i+5] = sign*Q[j][i];
        }
        if (gc_x >= 4 && ti == nx_ + gc_x - 4) {
            Q[j][i+7] = sign*Q[j][i];
        }
        if (gc_x >= 5 && ti == nx_ + gc_x - 5) {
            Q[j][i+9] = sign*Q[j][i];
        }
    }
}
    
    
// South boundary
template<int w, int h, int gc_x, int gc_y, int sign>
__device__ void bcSouthReflective(float Q[h+2*gc_y][w+2*gc_x], 
                                const int nx_, const int ny_) {
    for (int i=threadIdx.x; i<w+2*gc_x; i+=w) {
        const int j = threadIdx.y + gc_y;
        const int tj = blockDim.y*blockIdx.y + j;

        if (gc_y >= 1 && tj == gc_y) {
            Q[j-1][i] = sign*Q[j][i];
        }
        if (gc_y >= 2 && tj == gc_y + 1) {
            Q[j-3][i] = sign*Q[j][i];
        }
        if (gc_y >= 3 && tj == gc_y + 2) {
            Q[j-5][i] = sign*Q[j][i];
        }
        if (gc_y >= 4 && tj == gc_y + 3) {
            Q[j-7][i] = sign*Q[j][i];
        }
        if (gc_y >= 5 && tj == gc_y + 4) {
            Q[j-9][i] = sign*Q[j][i];
        }
    }
}



    
// North boundary
template<int w, int h, int gc_x, int gc_y, int sign>
__device__ void bcNorthReflective(float Q[h+2*gc_y][w+2*gc_x], const int nx_, const int ny_) {
    for (int i=threadIdx.x; i<w+2*gc_x; i+=w) {
        const int j = threadIdx.y + gc_y;
        const int tj = blockDim.y*blockIdx.y + j;
        
        if (gc_y >= 1 && tj == ny_ + gc_y - 1) {
            Q[j+1][i] = sign*Q[j][i];
        }
        if (gc_y >= 2 && tj == ny_ + gc_y - 2) {
            Q[j+3][i] = sign*Q[j][i];
        }
        if (gc_y >= 3 && tj == ny_ + gc_y - 3) {
            Q[j+5][i] = sign*Q[j][i];
        }
        if (gc_y >= 4 && tj == ny_ + gc_y - 4) {
            Q[j+7][i] = sign*Q[j][i];
        }
        if (gc_y >= 5 && tj == ny_ + gc_y - 5) {
            Q[j+9][i] = sign*Q[j][i];
        }
    }
}




/**
  * Alter the index l so that it gives periodic boundary conditions when reading
  */
template<int gc_x>
inline __device__ int handlePeriodicBoundaryX(int k, int nx_, int boundary_conditions_) {
    const int gc_pad = gc_x;
    
    //West boundary: add an offset to read from east of domain
    if (gc_x > 0) {
        if ((k < gc_pad) 
                && getBCWest(boundary_conditions_) == Periodic) {
            k += (nx_+2*gc_x - 2*gc_pad);
        }
        //East boundary: subtract an offset to read from west of domain
        else if ((k >= nx_+2*gc_x-gc_pad) 
                && getBCEast(boundary_conditions_) == Periodic) {
            k -= (nx_+2*gc_x - 2*gc_pad);
        }
    }
    
    return k;
}

/**
  * Alter the index l so that it gives periodic boundary conditions when reading
  */
template<int gc_y>
inline __device__ int handlePeriodicBoundaryY(int l, int ny_, int boundary_conditions_) {
    const int gc_pad = gc_y;
    
    //South boundary: add an offset to read from north of domain
    if (gc_y > 0) {
        if ((l < gc_pad) 
                && getBCSouth(boundary_conditions_) == Periodic) {
            l += (ny_+2*gc_y - 2*gc_pad);
        }
        //North boundary: subtract an offset to read from south of domain
        else if ((l >= ny_+2*gc_y-gc_pad) 
                && getBCNorth(boundary_conditions_) == Periodic) {
            l -= (ny_+2*gc_y - 2*gc_pad);
        }
    }
    
    return l;
}


template<int w, int h, int gc_x, int gc_y, int sign_x, int sign_y>
inline __device__ 
void handleReflectiveBoundary(
                float Q[h+2*gc_y][w+2*gc_x], 
                const int nx_, const int ny_,
                const int boundary_conditions_) {

    //Handle reflective boundary conditions
    if (getBCNorth(boundary_conditions_) == Reflective) {
        bcNorthReflective<w, h, gc_x, gc_y, sign_y>(Q, nx_, ny_);
        __syncthreads();
    }
    if (getBCSouth(boundary_conditions_) == Reflective) {
        bcSouthReflective<w, h, gc_x, gc_y, sign_y>(Q, nx_, ny_);
        __syncthreads();
    }
    if (getBCEast(boundary_conditions_) == Reflective) {
        bcEastReflective<w, h, gc_x, gc_y, sign_x>(Q, nx_, ny_);
        __syncthreads();
    }
    if (getBCWest(boundary_conditions_) == Reflective) {
        bcWestReflective<w, h, gc_x, gc_y, sign_x>(Q, nx_, ny_);
        __syncthreads();
    }
}

/**
  * Reads a block of data with ghost cells
  */
template<int w, int h, int gc_x, int gc_y, int sign_x, int sign_y>
inline __device__ void readBlock(float* ptr_, int pitch_,
                float Q[h+2*gc_y][w+2*gc_x], 
                const int nx_, const int ny_,
                const int boundary_conditions_,
                 int x0, int y0,
                 int x1, int y1) {
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Read into shared memory
    //Loop over all variables
    for (int j=threadIdx.y; j<h+2*gc_y; j+=h) {
        //Handle periodic boundary conditions here
        int l = handlePeriodicBoundaryY<gc_y>(by + j + y0, ny_, boundary_conditions_);
        l = min(l, min(ny_+2*gc_y-1, y1+2*gc_y-1));
        float* row = (float*) ((char*) ptr_ + pitch_*l);
        
        for (int i=threadIdx.x; i<w+2*gc_x; i+=w) {
            //Handle periodic boundary conditions here
            int k = handlePeriodicBoundaryX<gc_x>(bx + i + x0, nx_, boundary_conditions_);
            k = min(k, min(nx_+2*gc_x-1, x1+2*gc_x-1));
            
            //Read from global memory
            Q[j][i] = row[k];
        }
    }
    __syncthreads();
    
    handleReflectiveBoundary<w, h, gc_x, gc_y, sign_x, sign_y>(Q, nx_, ny_, boundary_conditions_);
}




/**
  * Writes a block of data to global memory for the shallow water equations.
  */
template<int w, int h, int gc_x, int gc_y>
inline __device__ void writeBlock(float* ptr_, int pitch_,
                 float shmem[h+2*gc_y][w+2*gc_x],
                 const int nx_, const int ny_,
                 int rk_step_, int rk_order_,
                 int x0, int y0,
                 int x1, int y1) {
    
    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + gc_x + x0;
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + gc_y + y0;

    //In case we are writing only to a subarea given by (x0, y0) x (x1, y1)
    const int max_ti = min(nx_+gc_x, x1+gc_x);
    const int max_tj = min(ny_+gc_y, y1+gc_y);
    
    //Only write internal cells
    if ((x0+gc_x <= ti) && (ti < max_ti) && (y0+gc_y <= tj) && (tj < max_tj)) {
        //Index of thread within block
        const int tx = threadIdx.x + gc_x;
        const int ty = threadIdx.y + gc_y;
        
        float* const row  = (float*) ((char*) ptr_ + pitch_*tj);
        
        //Handle runge-kutta timestepping here
        row[ti] = shmem[ty][tx];
        
        
        
        /**
          * SSPRK1 (forward Euler)
          * u^1   = u^n + dt*f(u^n)
          */
        if (rk_order_ == 1) {
            row[ti] = shmem[ty][tx];
        }
        /**
          * SSPRK2
          * u^1   = u^n + dt*f(u^n)
          * u^n+1 = 1/2*u^n + 1/2*(u^1 + dt*f(u^1))
          */
        else if (rk_order_ == 2) {
            if (rk_step_ == 0) {
                row[ti] = shmem[ty][tx];
            }
            else if (rk_step_ == 1) {
                row[ti] = 0.5f*row[ti] + 0.5f*shmem[ty][tx];
            }
        }
        /**
          * SSPRK3
          * u^1   = u^n + dt*f(u^n)
          * u^2   = 3/4 * u^n + 1/4 * (u^1 + dt*f(u^1))
          * u^n+1 = 1/3 * u^n + 2/3 * (u^2 + dt*f(u^2))
          * FIXME: This is not correct now, need a temporary to hold intermediate step u^2
          */
        else if (rk_order_ == 3) {
            if (rk_step_ == 0) {
                row[ti] = shmem[ty][tx];
            }
            else if (rk_step_ == 1) {
                row[ti] = 0.75f*row[ti] + 0.25f*shmem[ty][tx];
            }
            else if (rk_step_ == 2) {
                const float t = 1.0f / 3.0f; //Not representable in base 2
                row[ti] = t*row[ti] + (1.0f-t)*shmem[ty][tx];
            }
        }

        // DEBUG
        //row[ti] = 99.0;
    }
}











template<int w, int h, int gc_x, int gc_y, int vars>
__device__ void evolveF(float Q[vars][h+2*gc_y][w+2*gc_x],
              float F[vars][h+2*gc_y][w+2*gc_x],
              const float dx_, const float dt_) {
    for (int var=0; var < vars; ++var) {
        for (int j=threadIdx.y; j<h+2*gc_y; j+=h) {
            for (int i=threadIdx.x+gc_x; i<w+gc_x; i+=w) {
                Q[var][j][i] = Q[var][j][i] + (F[var][j][i-1] - F[var][j][i]) * dt_ / dx_;
            }
        }
    }
}






/**
  * Evolves the solution in time along the y axis (dimensional splitting)
  */
template<int w, int h, int gc_x, int gc_y, int vars>
__device__ void evolveG(float Q[vars][h+2*gc_y][w+2*gc_x],
              float G[vars][h+2*gc_y][w+2*gc_x],
              const float dy_, const float dt_) {
    for (int var=0; var < vars; ++var) {
        for (int j=threadIdx.y+gc_y; j<h+gc_y; j+=h) {
            for (int i=threadIdx.x; i<w+2*gc_x; i+=w) {
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





template <unsigned int threads>
__device__ void reduce_max(float* data, unsigned int n) {
	__shared__ float sdata[threads];
	unsigned int tid = threadIdx.x;

	//Reduce to "threads" elements
	sdata[tid] = FLT_MIN;
	for (unsigned int i=tid; i<n; i += threads) {
		sdata[tid] = max(sdata[tid], dt_ctx.L[i]);
    }
	__syncthreads();

	//Now, reduce all elements into a single element
	if (threads >= 512) {
		if (tid < 256) {
            sdata[tid] = max(sdata[tid], sdata[tid + 256]);
        }
		__syncthreads();
	}
	if (threads >= 256) {
		if (tid < 128) {
            sdata[tid] = max(sdata[tid], sdata[tid + 128]);
        }
		__syncthreads();
	}
	if (threads >= 128) {
		if (tid < 64) {
            sdata[tid] = max(sdata[tid], sdata[tid + 64]);
        }
		__syncthreads();
	}
	if (tid < 32) {
        volatile float* sdata_volatile = sdata;
		if (threads >= 64) {
            sdata_volatile[tid] = max(sdata_volatile[tid], sdata_volatile[tid + 32]);
        }
		if (tid < 16) {
			if (threads >= 32) sdata_volatile[tid] = max(sdata_volatile[tid], sdata_volatile[tid + 16]);
			if (threads >= 16) sdata_volatile[tid] = max(sdata_volatile[tid], sdata_volatile[tid +  8]);
			if (threads >=  8) sdata_volatile[tid] = max(sdata_volatile[tid], sdata_volatile[tid +  4]);
			if (threads >=  4) sdata_volatile[tid] = max(sdata_volatile[tid], sdata_volatile[tid +  2]);
			if (threads >=  2) sdata_volatile[tid] = max(sdata_volatile[tid], sdata_volatile[tid +  1]);
		}

		if (tid == 0) {
            return sdata_volatile[0];
		}
	}
}











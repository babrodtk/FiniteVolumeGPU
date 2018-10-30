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
  * Reads a block of data with ghost cells
  */
template<int block_width, int block_height, int ghost_cells>
inline __device__ void readBlock(float* ptr_, int pitch_,
                float shmem[block_height+2*ghost_cells][block_width+2*ghost_cells], 
                const int max_x_, const int max_y_) {
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;
        
    //Read into shared memory
    //Loop over all variables
    for (int j=threadIdx.y; j<block_height+2*ghost_cells; j+=block_height) {
        const int l = min(by + j, max_y_-1);
        float* row = (float*) ((char*) ptr_  + pitch_*l);
        
        for (int i=threadIdx.x; i<block_width+2*ghost_cells; i+=block_width) {
            const int k = min(bx + i, max_x_-1);
            
            shmem[j][i] = row[k];
        }
    }
}




/**
  * Writes a block of data to global memory for the shallow water equations.
  */
template<int block_width, int block_height, int ghost_cells>
inline __device__ void writeBlock(float* ptr_, int pitch_,
                 float shmem[block_height+2*ghost_cells][block_width+2*ghost_cells],
                 const int width, const int height) {
    
    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + ghost_cells;
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + ghost_cells;
    
    //Only write internal cells
    if (ti < width+ghost_cells && tj < height+ghost_cells) {
        //Index of thread within block
        const int tx = threadIdx.x + ghost_cells;
        const int ty = threadIdx.y + ghost_cells;
        
        float* const row  = (float*) ((char*) ptr_ + pitch_*tj);
        row[ti] = shmem[ty][tx];
    }
}













template<int block_width, int block_height, int ghost_cells, int scale_east_west=1, int scale_north_south=1>
__device__ void noFlowBoundary(float Q[block_height+2*ghost_cells][block_width+2*ghost_cells], const int nx_, const int ny_) {
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + ghost_cells;
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + ghost_cells;
    
    const int i = threadIdx.x + ghost_cells;
    const int j = threadIdx.y + ghost_cells;
    
    
    // West boundary
    if (ti == ghost_cells) {
        Q[j][i-1] = scale_east_west*Q[j][i];
    }
    if (ghost_cells >= 2 && ti == ghost_cells + 1) {
        Q[j][i-3] = scale_east_west*Q[j][i];
    }
    if (ghost_cells >= 3 && ti == ghost_cells + 2) {
        Q[j][i-5] = scale_east_west*Q[j][i];
    }
    if (ghost_cells >= 4 && ti == ghost_cells + 3) {
        Q[j][i-7] = scale_east_west*Q[j][i];
    }
    if (ghost_cells >= 5 && ti == ghost_cells + 4) {
        Q[j][i-9] = scale_east_west*Q[j][i];
    }
    
    
    
    // East boundary
    if (ti == nx_ + ghost_cells - 1) {
        Q[j][i+1] = scale_east_west*Q[j][i];
    }
    if (ghost_cells >= 2 && ti == nx_ + ghost_cells - 2) {
        Q[j][i+3] = scale_east_west*Q[j][i];
    }
    if (ghost_cells >= 3 && ti == nx_ + ghost_cells - 3) {
        Q[j][i+5] = scale_east_west*Q[j][i];
    }
    if (ghost_cells >= 3 && ti == nx_ + ghost_cells - 4) {
        Q[j][i+7] = scale_east_west*Q[j][i];
    }
    if (ghost_cells >= 3 && ti == nx_ + ghost_cells - 5) {
        Q[j][i+9] = scale_east_west*Q[j][i];
    }
    
    
    
    
    // South boundary
    if (tj == ghost_cells) {
        Q[j-1][i] = scale_north_south*Q[j][i];
    }
    if (ghost_cells >= 2 && tj == ghost_cells + 1) {
        Q[j-3][i] = scale_north_south*Q[j][i];
    }
    if (ghost_cells >= 3 && tj == ghost_cells + 2) {
        Q[j-5][i] = scale_north_south*Q[j][i];
    }
    if (ghost_cells >= 4 && tj == ghost_cells + 3) {
        Q[j-7][i] = scale_north_south*Q[j][i];
    }
    if (ghost_cells >= 5 && tj == ghost_cells + 4) {
        Q[j-9][i] = scale_north_south*Q[j][i];
    }
    
    
    
    // North boundary
    if (tj == ny_ + ghost_cells - 1) {
        Q[j+1][i] = scale_north_south*Q[j][i];
    }
    if (ghost_cells >= 2 && tj == ny_ + ghost_cells - 2) {
        Q[j+3][i] = scale_north_south*Q[j][i];
    }
    if (ghost_cells >= 3 && tj == ny_ + ghost_cells - 3) {
        Q[j+5][i] = scale_north_south*Q[j][i];
    }
    if (ghost_cells >= 3 && tj == ny_ + ghost_cells - 4) {
        Q[j+7][i] = scale_north_south*Q[j][i];
    }
    if (ghost_cells >= 3 && tj == ny_ + ghost_cells - 5) {
        Q[j+9][i] = scale_north_south*Q[j][i];
    }
}















/**
  * Evolves the solution in time along the x axis (dimensional splitting)
  */
__device__ void evolveF1(float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
              float F[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
              const int nx_, const int ny_,
              const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + 1; //Skip global ghost cells, i.e., +1
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + 1;
    
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;
        
        Q[0][j][i] = Q[0][j][i] + (F[0][ty][tx] - F[0][ty][tx+1]) * dt_ / dx_;
        Q[1][j][i] = Q[1][j][i] + (F[1][ty][tx] - F[1][ty][tx+1]) * dt_ / dx_;
        Q[2][j][i] = Q[2][j][i] + (F[2][ty][tx] - F[2][ty][tx+1]) * dt_ / dx_;
    }
}






/**
  * Evolves the solution in time along the x axis (dimensional splitting)
  */
__device__ void evolveF2(float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
              float F[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
              const int nx_, const int ny_,
              const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + 2;
    
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +1
        const int j = ty + 2;
        
        Q[0][j][i] = Q[0][j][i] + (F[0][ty][tx] - F[0][ty][tx+1]) * dt_ / dx_;
        Q[1][j][i] = Q[1][j][i] + (F[1][ty][tx] - F[1][ty][tx+1]) * dt_ / dx_;
        Q[2][j][i] = Q[2][j][i] + (F[2][ty][tx] - F[2][ty][tx+1]) * dt_ / dx_;
    }
}






/**
  * Evolves the solution in time along the y axis (dimensional splitting)
  */
__device__ void evolveG1(float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
              float G[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
              const int nx_, const int ny_,
              const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + 1; //Skip global ghost cells, i.e., +1
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + 1;
    
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;
        
        Q[0][j][i] = Q[0][j][i] + (G[0][ty][tx] - G[0][ty+1][tx]) * dt_ / dy_;
        Q[1][j][i] = Q[1][j][i] + (G[1][ty][tx] - G[1][ty+1][tx]) * dt_ / dy_;
        Q[2][j][i] = Q[2][j][i] + (G[2][ty][tx] - G[2][ty+1][tx]) * dt_ / dy_;
    }
}







/**
  * Evolves the solution in time along the y axis (dimensional splitting)
  */
__device__ void evolveG2(float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
              float G[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
              const int nx_, const int ny_,
              const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + 2;
    
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;
        
        Q[0][j][i] = Q[0][j][i] + (G[0][ty][tx] - G[0][ty+1][tx]) * dt_ / dy_;
        Q[1][j][i] = Q[1][j][i] + (G[1][ty][tx] - G[1][ty+1][tx]) * dt_ / dy_;
        Q[2][j][i] = Q[2][j][i] + (G[2][ty][tx] - G[2][ty+1][tx]) * dt_ / dy_;
    }
}



























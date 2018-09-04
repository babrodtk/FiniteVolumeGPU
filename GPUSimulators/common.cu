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




/**
  * Reads a block of data  with one ghost cell for the shallow water equations
  */
template<int sm_width, int sm_height, int block_width, int block_height>
inline __device__ void readBlock(float* ptr_, int pitch_,
                float (&shmem)[sm_height][sm_width], 
                const int max_x_, const int max_y_) {
    
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;
    
    //Read into shared memory
    for (int j=threadIdx.y; j<sm_height; j+=block_height) {
        const int l = clamp(by + j, 0, max_y_-1); // Clamp out of bounds
        
        //Compute the pointer to current row in the arrays
        const float* const row  = (float*) ((char*) ptr_  + pitch_*l);
        
        for (int i=threadIdx.x; i<sm_width; i+=block_width) {
            const int k = clamp(bx + i, 0, max_x_-1); // Clamp out of bounds

            shmem[j][i] = row[k];
        }
    }
}


/**
  * Reads a block of data with ghost cells
  */
template<int vars, int sm_width, int sm_height, int block_width, int block_height>
inline __device__ void readBlock(float* ptr_[vars], int pitch_[vars],
                float shmem[vars][sm_height][sm_width], 
                const int max_x_, const int max_y_) {
    
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;
    
    float* rows[3];
    
    //Read into shared memory
    for (int j=threadIdx.y; j<sm_height; j+=block_height) {
        const int l = clamp(by + j, 0, max_y_-1); // Clamp out of bounds
        
        //Compute the pointer to current row in the arrays
        for (int m=0; m<vars; ++m) {
            rows[m] = (float*) ((char*) ptr_[m]  + pitch_[m]*l);
        }
        
        for (int i=threadIdx.x; i<sm_width; i+=block_width) {
            const int k = clamp(bx + i, 0, max_x_-1); // Clamp out of bounds

            for (int m=0; m<vars; ++m) {
                shmem[m][j][i] = rows[m][k];
            }
        }
    }
}





/**
  * Writes a block of data to global memory for the shallow water equations.
  */
template<int sm_width, int sm_height, int offset_x=0, int offset_y=0>
inline __device__ void writeBlock(float* ptr_, int pitch_,
                 float shmem[sm_height][sm_width],
                 const int width, const int height) {
    
    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + offset_x;
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + offset_y;
    
    //Only write internal cells
    if (ti < width+offset_x && tj < height+offset_y) {
        //Index of thread within block
        const int tx = threadIdx.x + offset_x;
        const int ty = threadIdx.y + offset_y;
        
        float* const row  = (float*) ((char*) ptr_ + pitch_*tj);
        row[ti] = shmem[ty][tx];
    }
}








/**
  * Writes a block of data to global memory for the shallow water equations.
  */
__device__ void writeBlock2(float* h_ptr_, int h_pitch_,
                 float* hu_ptr_, int hu_pitch_,
                 float* hv_ptr_, int hv_pitch_,
                 float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4], 
                 const int nx_, const int ny_) {
    writeBlock<BLOCK_WIDTH+4, BLOCK_HEIGHT+4, 2, 2>( h_ptr_,  h_pitch_, Q[0], nx_, ny_);
    writeBlock<BLOCK_WIDTH+4, BLOCK_HEIGHT+4, 2, 2>(hu_ptr_, hu_pitch_, Q[1], nx_, ny_);
    writeBlock<BLOCK_WIDTH+4, BLOCK_HEIGHT+4, 2, 2>(hv_ptr_, hv_pitch_, Q[2], nx_, ny_);
}






/**
  * No flow boundary conditions for the shallow water equations
  * with one ghost cell in each direction
  */
__device__ void noFlowBoundary1(float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2], const int nx_, const int ny_) {
    //Global index
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + 1; //Skip global ghost cells, i.e., +1
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + 1;
    
    //Block-local indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int i = tx + 1; //Skip local ghost cells, i.e., +1
    const int j = ty + 1;
    
    //Fix boundary conditions
    if (ti == 1) {
        Q[0][j][i-1] =  Q[0][j][i];
        Q[1][j][i-1] = -Q[1][j][i];
        Q[2][j][i-1] =  Q[2][j][i];
    }
    if (ti == nx_) {
        Q[0][j][i+1] =  Q[0][j][i];
        Q[1][j][i+1] = -Q[1][j][i];
        Q[2][j][i+1] =  Q[2][j][i];
    }
    if (tj == 1) {
        Q[0][j-1][i] =  Q[0][j][i];
        Q[1][j-1][i] =  Q[1][j][i];
        Q[2][j-1][i] = -Q[2][j][i];
    }
    if (tj == ny_) {
        Q[0][j+1][i] =  Q[0][j][i];
        Q[1][j+1][i] =  Q[1][j][i];
        Q[2][j+1][i] = -Q[2][j][i];
    }
}






/**
  * No flow boundary conditions for the shallow water equations
  * with two ghost cells in each direction
  */
__device__ void noFlowBoundary2(float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4], const int nx_, const int ny_) {
    //Global index
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + 2;
    
    //Block-local indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int i = tx + 2; //Skip local ghost cells, i.e., +2
    const int j = ty + 2;
    
    if (ti == 2) {
        Q[0][j][i-1] =  Q[0][j][i];
        Q[1][j][i-1] = -Q[1][j][i];
        Q[2][j][i-1] =  Q[2][j][i];
        
        Q[0][j][i-2] =  Q[0][j][i+1];
        Q[1][j][i-2] = -Q[1][j][i+1];
        Q[2][j][i-2] =  Q[2][j][i+1];
    }
    if (ti == nx_+1) {
        Q[0][j][i+1] =  Q[0][j][i];
        Q[1][j][i+1] = -Q[1][j][i];
        Q[2][j][i+1] =  Q[2][j][i];
        
        Q[0][j][i+2] =  Q[0][j][i-1];
        Q[1][j][i+2] = -Q[1][j][i-1];
        Q[2][j][i+2] =  Q[2][j][i-1];
    }
    if (tj == 2) {
        Q[0][j-1][i] =  Q[0][j][i];
        Q[1][j-1][i] =  Q[1][j][i];
        Q[2][j-1][i] = -Q[2][j][i];
        
        Q[0][j-2][i] =  Q[0][j+1][i];
        Q[1][j-2][i] =  Q[1][j+1][i];
        Q[2][j-2][i] = -Q[2][j+1][i];
    }
    if (tj == ny_+1) {
        Q[0][j+1][i] =  Q[0][j][i];
        Q[1][j+1][i] =  Q[1][j][i];
        Q[2][j+1][i] = -Q[2][j][i];
        
        Q[0][j+2][i] =  Q[0][j-1][i];
        Q[1][j+2][i] =  Q[1][j-1][i];
        Q[2][j+2][i] = -Q[2][j-1][i];
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




























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
  * Location of thread in block
  */
inline __device__ int get_local_id(int dim) {
    switch(dim) {
        case 0: return threadIdx.x; 
        case 1: return threadIdx.y;
        case 2: return threadIdx.z;
        default: return -1;
    }
}


/**
  * Get block index
  */
__device__ int get_group_id(int dim) {
    switch(dim) {
        case 0: return blockIdx.x;
        case 1: return blockIdx.y;
        case 2: return blockIdx.z;
        default: return -1;
    }
}

/**
  * Location of thread in global domain
  */
__device__ int get_global_id(int dim) {
    switch(dim) {
        case 0: return blockDim.x*blockIdx.x + threadIdx.x;
        case 1: return blockDim.y*blockIdx.y + threadIdx.y;
        case 2: return blockDim.z*blockIdx.z + threadIdx.z;
        default: return -1;
    }
}


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
__device__ void readBlock1(float* h_ptr_, int h_pitch_,
                float* hu_ptr_, int hu_pitch_,
                float* hv_ptr_, int hv_pitch_,
                float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2], 
                const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = BLOCK_WIDTH * get_group_id(0);
    const int by = BLOCK_HEIGHT * get_group_id(1);
    
    //Read into shared memory
    for (int j=ty; j<BLOCK_HEIGHT+2; j+=BLOCK_HEIGHT) {
        const int l = clamp(by + j, 0, ny_+1); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        float* const h_row  = (float*) ((char*) h_ptr_  + h_pitch_*l);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*l);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*l);
        
        for (int i=tx; i<BLOCK_WIDTH+2; i+=BLOCK_WIDTH) {
            const int k = clamp(bx + i, 0, nx_+1); // Out of bounds
            
            Q[0][j][i] = h_row[k];
            Q[1][j][i] = hu_row[k];
            Q[2][j][i] = hv_row[k];
        }
    }
}





/**
  * Reads a block of data  with two ghost cells for the shallow water equations
  */
__device__ void readBlock2(float* h_ptr_, int h_pitch_,
                float* hu_ptr_, int hu_pitch_,
                float* hv_ptr_, int hv_pitch_,
                float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4], 
                const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = BLOCK_WIDTH * get_group_id(0);
    const int by = BLOCK_HEIGHT * get_group_id(1);
    
    //Read into shared memory
    for (int j=ty; j<BLOCK_HEIGHT+4; j+=BLOCK_HEIGHT) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        float* const h_row  = (float*) ((char*) h_ptr_  + h_pitch_*l);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*l);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*l);
        
        for (int i=tx; i<BLOCK_WIDTH+4; i+=BLOCK_WIDTH) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds
            
            Q[0][j][i] = h_row[k];
            Q[1][j][i] = hu_row[k];
            Q[2][j][i] = hv_row[k];
        }
    }
}




/**
  * Writes a block of data to global memory for the shallow water equations.
  */
__device__ void writeBlock1(float* h_ptr_, int h_pitch_,
                 float* hu_ptr_, int hu_pitch_,
                 float* hv_ptr_, int hv_pitch_,
                 float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
                 const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
    //Only write internal cells
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;

        float* const h_row  = (float*) ((char*) h_ptr_  + h_pitch_*tj);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*tj);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*tj);
        
        h_row[ti]  = Q[0][j][i];
        hu_row[ti] = Q[1][j][i];
        hv_row[ti] = Q[2][j][i];
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
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    //Only write internal cells
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;

        float* const h_row  = (float*) ((char*) h_ptr_ + h_pitch_*tj);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*tj);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*tj);
        
        h_row[ti]  = Q[0][j][i];
        hu_row[ti] = Q[1][j][i];
        hv_row[ti] = Q[2][j][i];
    }
}






/**
  * No flow boundary conditions for the shallow water equations
  * with one ghost cell in each direction
  */
__device__ void noFlowBoundary1(float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2], const int nx_, const int ny_) {
    //Global index
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
    //Block-local indices
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
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
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    //Block-local indices
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
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
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
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
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
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
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
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
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;
        
        Q[0][j][i] = Q[0][j][i] + (G[0][ty][tx] - G[0][ty+1][tx]) * dt_ / dy_;
        Q[1][j][i] = Q[1][j][i] + (G[1][ty][tx] - G[1][ty+1][tx]) * dt_ / dy_;
        Q[2][j][i] = Q[2][j][i] + (G[2][ty][tx] - G[2][ty+1][tx]) * dt_ / dy_;
    }
}










__device__ float3 F_func(const float3 Q, const float g) {
    float3 F;

    F.x = Q.y;                              //hu
    F.y = Q.y*Q.y / Q.x + 0.5f*g*Q.x*Q.x;   //hu*hu/h + 0.5f*g*h*h;
    F.z = Q.y*Q.z / Q.x;                    //hu*hv/h;

    return F;
}


























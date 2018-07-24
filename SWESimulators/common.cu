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


__device__ int get_local_size(int dim) {
    switch(dim) {
        case 0: return blockDim.x;
        case 1: return blockDim.y;
        case 2: return blockDim.z;
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
                float Q[3][block_height+2][block_width+2], 
                const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);
    
    //Read into shared memory
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+1); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        float* const h_row  = (float*) ((char*) h_ptr_  + h_pitch_*l);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*l);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*l);
        
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
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
                float Q[3][block_height+4][block_width+4], 
                const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);
    
    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        float* const h_row  = (float*) ((char*) h_ptr_  + h_pitch_*l);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*l);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*l);
        
        for (int i=tx; i<block_width+4; i+=get_local_size(0)) {
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
                 float Q[3][block_height+2][block_width+2],
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
                 float Q[3][block_height+4][block_width+4], 
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
__device__ void noFlowBoundary1(float Q[3][block_height+2][block_width+2], const int nx_, const int ny_) {
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
__device__ void noFlowBoundary2(float Q[3][block_height+4][block_width+4], const int nx_, const int ny_) {
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
__device__ void evolveF1(float Q[3][block_height+2][block_width+2],
              float F[3][block_height+1][block_width+1],
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
__device__ void evolveF2(float Q[3][block_height+4][block_width+4],
              float F[3][block_height+1][block_width+1],
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
__device__ void evolveG1(float Q[3][block_height+2][block_width+2],
              float G[3][block_height+1][block_width+1],
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
__device__ void evolveG2(float Q[3][block_height+4][block_width+4],
              float G[3][block_height+1][block_width+1],
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










/**
  * Reconstructs a slope using the minmod limiter based on three 
  * consecutive values
  */
__device__ float minmodSlope(float left, float center, float right, float theta) {
    const float backward = (center - left) * theta;
    const float central = (right - left) * 0.5f;
    const float forward = (right - center) * theta;
    
	return 0.25f
		*copysign(1.0f, backward)
		*(copysign(1.0f, backward) + copysign(1.0f, central))
		*(copysign(1.0f, central) + copysign(1.0f, forward))
		*min( min(fabs(backward), fabs(central)), fabs(forward) );
}




/**
  * Reconstructs a minmod slope for a whole block along x
  */
__device__ void minmodSlopeX(float  Q[3][block_height+4][block_width+4],
                  float Qx[3][block_height+2][block_width+2],
                  const float theta_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Reconstruct slopes along x axis
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            const int k = i + 1;
            for (int p=0; p<3; ++p) {
                Qx[p][j][i] = minmodSlope(Q[p][l][k-1], Q[p][l][k], Q[p][l][k+1], theta_);
            }
        }
    }
}


/**
  * Reconstructs a minmod slope for a whole block along y
  */
__device__ void minmodSlopeY(float  Q[3][block_height+4][block_width+4],
                  float Qy[3][block_height+2][block_width+2],
                  const float theta_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 2; //Skip ghost cells
            for (int p=0; p<3; ++p) {
                Qy[p][j][i] = minmodSlope(Q[p][l-1][k], Q[p][l][k], Q[p][l+1][k], theta_);
            }
        }
    }
}









__device__ float3 F_func(const float3 Q, const float g) {
    float3 F;

    F.x = Q.y;                              //hu
    F.y = Q.y*Q.y / Q.x + 0.5f*g*Q.x*Q.x;   //hu*hu/h + 0.5f*g*h*h;
    F.z = Q.y*Q.z / Q.x;                    //hu*hv/h;

    return F;
}





/**
  * Central upwind flux function
  */
__device__ float3 CentralUpwindFlux(const float3 Qm, float3 Qp, const float g) {
    const float3 Fp = F_func(Qp, g);
    const float up = Qp.y / Qp.x;   // hu / h
    const float cp = sqrt(g*Qp.x); // sqrt(g*h)

    const float3 Fm = F_func(Qm, g);
    const float um = Qm.y / Qm.x;   // hu / h
    const float cm = sqrt(g*Qm.x); // sqrt(g*h)
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed
    
    return ((ap*Fm - am*Fp) + ap*am*(Qp-Qm))/(ap-am);
}










/**
  * Harten-Lax-van Leer with contact discontinuity (Toro 2001, p 180)
  */
__device__ float3 HLL_flux(const float3 Q_l, const float3 Q_r, const float g_) {    
    const float h_l = Q_l.x;
    const float h_r = Q_r.x;
    
    // Calculate velocities
    const float u_l = Q_l.y / h_l;
    const float u_r = Q_r.y / h_r;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    // Compute h in the "star region", h^dagger
    const float h_dag = 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag / (h_l*h_l) ) );
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag / (h_r*h_r) ) );
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l;
    const float S_r = u_r + c_r*q_r;
    
    //Upwind selection
    if (S_l >= 0.0f) {
        return F_func(Q_l, g_);
    }
    else if (S_r <= 0.0f) {
        return F_func(Q_r, g_);
    }
    //Or estimate flux in the star region
    else {
        const float3 F_l = F_func(Q_l, g_);
        const float3 F_r = F_func(Q_r, g_);
        const float3 flux = (S_r*F_l - S_l*F_r + S_r*S_l*(Q_r - Q_l)) / (S_r-S_l);
        return flux;
    }
}









/**
  * Harten-Lax-van Leer with contact discontinuity (Toro 2001, p 181)
  */
__device__ float3 HLLC_flux(const float3 Q_l, const float3 Q_r, const float g_) {    
    const float h_l = Q_l.x;
    const float h_r = Q_r.x;
    
    // Calculate velocities
    const float u_l = Q_l.y / h_l;
    const float u_r = Q_r.y / h_r;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    // Compute h in the "star region", h^dagger
    const float h_dag = 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag / (h_l*h_l) ) );
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag / (h_r*h_r) ) );
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l;
    const float S_r = u_r + c_r*q_r;
    const float S_star = ( S_l*h_r*(u_r - S_r) - S_r*h_l*(u_l - S_l) ) / ( h_r*(u_r - S_r) - h_l*(u_l - S_l) );
    
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    //Upwind selection
    if (S_l >= 0.0f) {
        return F_l;
    }
    else if (S_r <= 0.0f) {
        return F_r;
    }
    //Or estimate flux in the "left star" region
    else if (S_l <= 0.0f && 0.0f <=S_star) {
        const float v_l = Q_l.z / h_l;
        const float3 Q_star_l = h_l * (S_l - u_l) / (S_l - S_star) * make_float3(1, S_star, v_l);
        const float3 flux = F_l + S_l*(Q_star_l - Q_l);
        return flux;
    }
    //Or estimate flux in the "righ star" region
    else if (S_star <= 0.0f && 0.0f <=S_r) {
        const float v_r = Q_r.z / h_r;
        const float3 Q_star_r = h_r * (S_r - u_r) / (S_r - S_star) * make_float3(1, S_star, v_r);
        const float3 flux = F_r + S_r*(Q_star_r - Q_r);
        return flux;
    }
    else {
        return make_float3(-99999.9f, -99999.9f, -99999.9f); //Something wrong here
    }
}



/**
  * Superbee flux limiter for WAF.
  * Related to superbee limiter so that WAF_superbee(r, c) = 1 - (1-|c|)*superbee(r)
  * @param r_ the ratio of upwind change (see Toro 2001, p. 203/204)
  * @param c_ the courant number for wave k, dt*S_k/dx
  */
__device__ float WAF_superbee(float r_, float c_) {
    // r <= 0.0
    if (r_ <= 0.0f) { 
        return 1.0f;
    }
    // 0.0 <= r <= 1/2
    else if (r_ <= 0.5f) { 
        return 1.0f - 2.0f*(1.0f - fabs(c_))*r_;
    }
    // 1/2 <= r <= 1
    else if (r_ <= 1.0f) {
        return fabs(c_);
    }
    // 1 <= r <= 2
    else  if (r_ <= 2.0f) {
        return 1.0f - (1.0f - fabs(c_))*r_;
    }
    // r >= 2
    else {
        return 2.0f*fabs(c_) - 1.0f;
    }
}




__device__ float WAF_albada(float r_, float c_) {
    if (r_ <= 0.0f) {
        return 1.0f;
    }
    else {
        return 1.0f - (1.0f - fabs(c_)) * r_ * (1.0f + r_) / (1.0f + r_*r_);
    }
}

__device__ float WAF_minmod(float r_, float c_) {
    return 1.0f - (1.0f - fabs(c_)) * fmax(0.0f, fmin(1.0f, r_));
}

__device__ float minmod(float r_) {
    return fmax(0.0f, fmin(1.0f, r_));
}

__device__ float superbee(float r_) {
    return fmax(0.0f, fmax(fmin(2.0f*r_, 1.0f), fmin(r_, 2.0f)));
}

__device__ float vanAlbada1(float r_) {
    return (r_*r_ + r_) / (r_*r_ + 1.0f);
}

__device__ float vanLeer(float r_) {
    return (r_ + fabs(r_)) / (1.0f + fabs(r_));
}

__device__ float limiterToWAFLimiter(float r_, float c_) {
    return 1.0f - (1.0f - fabs(c_))*r_;
}


/**
  * Weighted average flux (Toro 2001, p 200) for interface {i+1/2}
  * @param r_ The flux limiter parameter (see Toro 2001, p. 203)
  * @param Q_l2 Q_{i-1}
  * @param Q_l1 Q_{i}
  * @param Q_r1 Q_{i+1}
  * @param Q_r2 Q_{i+2}
  */
__device__ float3 WAF_1D_flux(const float3 Q_l2, const float3 Q_l1, const float3 Q_r1, const float3 Q_r2, const float g_, const float dx_, const float dt_) {     
    const float h_l = Q_l1.x;
    const float h_r = Q_r1.x;
    
    const float h_l2 = Q_l2.x;
    const float h_r2 = Q_r2.x;
    
    // Calculate velocities
    const float u_l = Q_l1.y / h_l;
    const float u_r = Q_r1.y / h_r;
    
    const float v_l = Q_l1.z / h_l;
    const float v_r = Q_r1.z / h_r;
    
    const float v_l2 = Q_l2.z / h_l2;
    const float v_r2 = Q_r2.z / h_r2;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    // Compute h in the "star region", h^dagger
    const float h_dag = 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag / (h_l*h_l) ) );
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag / (h_r*h_r) ) );
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l; //FIXME: Right wave speed estimate?
    const float S_r = u_r + c_r*q_r;
    const float S_star = ( S_l*h_r*(u_r - S_r) - S_r*h_l*(u_l - S_l) ) / ( h_r*(u_r - S_r) - h_l*(u_l - S_l) );
    
    const float3 Q_star_l = h_l * (S_l - u_l) / (S_l - S_star) * make_float3(1, S_star, v_l);
    const float3 Q_star_r = h_r * (S_r - u_r) / (S_r - S_star) * make_float3(1, S_star, v_r);
    
    // Estimate the fluxes in the four regions
    const float3 F_1 = F_func(Q_l1, g_);
    const float3 F_4 = F_func(Q_r1, g_);
    
    const float3 F_2 = F_1 + S_l*(Q_star_l - Q_l1);
    const float3 F_3 = F_4 + S_r*(Q_star_r - Q_r1);
    //const float3 F_2 = F_func(Q_star_l, g_);
    //const float3 F_3 = F_func(Q_star_r, g_);
    
    // Compute the courant numbers for the waves
    const float c_1 = S_l * dt_ / dx_;
    const float c_2 = S_star * dt_ / dx_;
    const float c_3 = S_r * dt_ / dx_;
    
    // Compute the "upwind change" vectors for the i-3/2 and i+3/2 interfaces
    const float rh_m = (h_l - h_l2) / (h_r - h_l);
    const float rh_p = (h_r2 - h_r) / (h_r - h_l);
    
    const float rv_m = (v_l - v_l2) / (v_r - v_l);
    const float rv_p = (v_r2 - v_r) / (v_r - v_l);
    
    // Compute the r parameters for the flux limiter
    const float rh_1 = (c_1 > 0.0f) ? rh_m : rh_p; 
    const float rv_1 = (c_1 > 0.0f) ? rv_m : rv_p; 
    
    const float rh_2 = (c_2 > 0.0f) ? rh_m : rh_p; 
    const float rv_2 = (c_2 > 0.0f) ? rv_m : rv_p; 
    
    const float rh_3 = (c_3 > 0.0f) ? rh_m : rh_p;
    const float rv_3 = (c_3 > 0.0f) ? rv_m : rv_p;
    
    // Compute the limiter
    // We use h for the nonlinear waves, and v for the middle shear wave 
    ///**
    const float A_1 = copysign(1.0f, c_1) * WAF_minmod(rh_1, c_1);
    const float A_2 = copysign(1.0f, c_2) * WAF_minmod(rv_2, c_2); //Middle shear wave 
    const float A_3 = copysign(1.0f, c_3) * WAF_minmod(rh_3, c_3); 
    //*/
    /**
    //2nd order for smooth cases (unstable for shocks)
    const float A_1 = c_1;
    const float A_2 = c_2;
    const float A_3 = c_3;
    */
    /*
    const float A_1 = sign(c_1) * limiterToWAFLimiter(minmod(rh_1), c_1);
    const float A_2 = sign(c_2) * limiterToWAFLimiter(minmod(rv_2), c_2);
    const float A_3 = sign(c_3) * limiterToWAFLimiter(minmod(rh_3), c_3);
    */
        
    //Average the fluxes
    const float3 flux = 0.5f*( F_1 + F_4 )
                      - 0.5f*( A_1 * (F_2 - F_1)
                             + A_2 * (F_3 - F_2)
                             + A_3 * (F_4 - F_3) );

    /*
    const float d_0 = -1.0f;
    const float d_1 = -0.5f;//max(min(sign(c_1)*WAF_minbee(rh_1, c_1), 1.0f), -1.0f);
    const float d_2 = 0.0f;//max(min(sign(c_2)*WAF_minbee(rh_2, c_2), 1.0f), -1.0f);
    const float d_3 = 0.5f;//max(min(sign(c_3)*WAF_minbee(rh_3, c_3), 1.0f), -1.0f);
    const float d_4 = 1.0f;
    const float3 flux = 0.5f*(d_1 - d_0) * F_1
                        + 0.5f*(d_2 - d_1) * F_2
                        + 0.5f*(d_3 - d_2) * F_3
                        + 0.5f*(d_4 - d_3) * F_4;
    */
    /*
    const float3 F_hllc = (S_r*F_1 - S_l*F_4 + S_r*S_l*(Q_r1 - Q_l1)) / (S_r-S_l);
    const float3 flux = 0.5f*(d_1 - d_0) * F_1
                        + 0.5f*(d_3 - d_1) * F_hllc
                        + 0.5f*(d_4 - d_3) * F_4;
      */
                             /*
    const float c_0 = -1.0f;
    const float c_4 = 1.0f;
    const float3 flux = 0.5f*(c_1 - c_0) * F_1
                        + 0.5f*(c_2 - c_1) * F_2
                        + 0.5f*(c_3 - c_2) * F_3
                        + 0.5f*(c_4 - c_3) * F_4;
                        */
    //const float3 flux = 0.5f*( F_1 + F_4 ) - 0.5f*( sign(c_3) * A_3 * (F_4 - F_3) );
    return flux;
}







/**
  * Lax-Friedrichs flux (Toro 2001, p 163)
  */
__device__ float3 LxF_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    return 0.5f*(F_l + F_r) + (dx_/(2.0f*dt_))*(Q_l - Q_r);
}



/**
  * Lax-Friedrichs extended to 2D
  */
__device__ float3 LxF_2D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    //Note numerical diffusion for 2D here (0.25)
    return 0.5f*(F_l + F_r) + (dx_/(4.0f*dt_))*(Q_l - Q_r);
}




/**
  * Richtmeyer / Two-step Lax-Wendroff flux (Toro 2001, p 164)
  */
__device__ float3 LxW2_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    const float3 Q_lw2 = 0.5f*(Q_l + Q_r) + (dt_/(2.0f*dx_))*(F_l - F_r);
    
    return F_func(Q_lw2, g_);
}






/**
  * Godunovs centered scheme (Toro 2001, p 165)
  */
__device__ float3 GodC_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    const float3 Q_godc = 0.5f*(Q_l + Q_r) + (dt_/dx_)*(F_l - F_r);
    
    return F_func(Q_godc, g_);
}
    

    
    
/**
  * First Ordered Centered (Toro 2001, p.163)
  */
__device__ float3 FORCE_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_lf = LxF_1D_flux(Q_l, Q_r, g_, dx_, dt_);
    const float3 F_lw2 = LxW2_1D_flux(Q_l, Q_r, g_, dx_, dt_);
    return 0.5f*(F_lf + F_lw2);
}






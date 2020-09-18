/*
This OpenCL kernel implements the classical Lax-Friedrichs scheme
for the shallow water equations, with edge fluxes.

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


#include "common.h"
#include "SWECommon.h"


/**
  * Computes the flux along the x axis for all faces
  */
__device__ 
void computeFluxF(float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
                  float F[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
                  const float g_, const float dx_, const float dt_) {
    //Compute fluxes along the x axis
    for (int j=threadIdx.y; j<BLOCK_HEIGHT+2; j+=BLOCK_HEIGHT) {
        for (int i=threadIdx.x; i<BLOCK_WIDTH+1; i+=BLOCK_WIDTH) {
            // Q at interface from the right and left
            const float3 Qp = make_float3(Q[0][j][i+1],
                                          Q[1][j][i+1],
                                          Q[2][j][i+1]);
            const float3 Qm = make_float3(Q[0][j][i],
                                          Q[1][j][i],
                                          Q[2][j][i]);
                                       
            // Computed flux
            const float3 flux = FORCE_1D_flux(Qm, Qp, g_, dx_, dt_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }
}


/**
  * Computes the flux along the y axis for all faces
  */
__device__ 
void computeFluxG(float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
                  float G[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
                  const float g_, const float dy_, const float dt_) {
    //Compute fluxes along the y axis
    for (int j=threadIdx.y; j<BLOCK_HEIGHT+1; j+=BLOCK_HEIGHT) {
        for (int i=threadIdx.x; i<BLOCK_WIDTH+2; i+=BLOCK_WIDTH) {
            // Q at interface from the right and left
            // Note that we swap hu and hv
            const float3 Qp = make_float3(Q[0][j+1][i],
                                          Q[2][j+1][i],
                                          Q[1][j+1][i]);
            const float3 Qm = make_float3(Q[0][j][i],
                                          Q[2][j][i],
                                          Q[1][j][i]);

            // Computed flux
            // Note that we swap back
            const float3 flux = FORCE_1D_flux(Qm, Qp, g_, dy_, dt_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
}


extern "C" {
__global__ void FORCEKernel(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
        int boundary_conditions_,
        
        //Input h^n
        float* h0_ptr_, int h0_pitch_,
        float* hu0_ptr_, int hu0_pitch_,
        float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        float* h1_ptr_, int h1_pitch_,
        float* hu1_ptr_, int hu1_pitch_,
        float* hv1_ptr_, int hv1_pitch_,
        
        //Output CFL
        float* cfl_) {
    
    const unsigned int w = BLOCK_WIDTH;
    const unsigned int h = BLOCK_HEIGHT;
    const unsigned int gc_x = 1;
    const unsigned int gc_y = 1;
    const unsigned int vars = 3;
    
    __shared__ float Q[vars][h+2*gc_y][w+2*gc_x];
    __shared__ float F[vars][h+2*gc_y][w+2*gc_x];
    
    //Read into shared memory
    readBlock<w, h, gc_x, gc_y,  1,  1>( h0_ptr_,  h0_pitch_, Q[0], nx_, ny_, boundary_conditions_);
    readBlock<w, h, gc_x, gc_y, -1,  1>(hu0_ptr_, hu0_pitch_, Q[1], nx_, ny_, boundary_conditions_);
    readBlock<w, h, gc_x, gc_y,  1, -1>(hv0_ptr_, hv0_pitch_, Q[2], nx_, ny_, boundary_conditions_);
    __syncthreads();
    
    //Compute flux along x, and evolve
    computeFluxF(Q, F, g_, dx_, dt_);
    __syncthreads();
    evolveF<w, h, gc_x, gc_y, vars>(Q, F, dx_, dt_);
    __syncthreads();
    
    //Compute flux along y, and evolve
    computeFluxG(Q, F, g_, dy_, dt_);
    __syncthreads();
    evolveG<w, h, gc_x, gc_y, vars>(Q, F, dy_, dt_);
    __syncthreads();
    
    //Write to main memory
    writeBlock<w, h, gc_x, gc_y>( h1_ptr_,  h1_pitch_, Q[0], nx_, ny_, 0, 1);
    writeBlock<w, h, gc_x, gc_y>(hu1_ptr_, hu1_pitch_, Q[1], nx_, ny_, 0, 1);
    writeBlock<w, h, gc_x, gc_y>(hv1_ptr_, hv1_pitch_, Q[2], nx_, ny_, 0, 1);
    
    //Compute the CFL for this block
    if (cfl_ != NULL) {
        writeCfl<w, h, gc_x, gc_y, vars>(Q, F[0], nx_, ny_, dx_, dy_, g_, cfl_);
    }
}

} // extern "C"
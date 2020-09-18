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
template <int block_width, int block_height>
__device__ 
void computeFluxF(float Q[3][block_height+2][block_width+2],
                  float F[3][block_height][block_width+1],
                  const float g_, const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    {
        const int j=ty;
        const int l = j + 1; //Skip ghost cells
        for (int i=tx; i<block_width+1; i+=block_width) {
            const int k = i;
            
            // Q at interface from the right and left
            const float3 Qp = make_float3(Q[0][l][k+1],
                                          Q[1][l][k+1],
                                          Q[2][l][k+1]);
            const float3 Qm = make_float3(Q[0][l][k],
                                          Q[1][l][k],
                                          Q[2][l][k]);
                                       
            // Computed flux
            const float3 flux = LxF_2D_flux(Qm, Qp, g_, dx_, dt_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }
}


/**
  * Computes the flux along the y axis for all faces
  */ 
template <int block_width, int block_height>
__device__
void computeFluxG(float Q[3][block_height+2][block_width+2],
                  float G[3][block_height+1][block_width],
                  const float g_, const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    for (int j=ty; j<block_height+1; j+=block_height) {
        const int l = j;
        {
            const int i=tx;
            const int k = i + 1; //Skip ghost cells
            
            // Q at interface from the right and left
            // Note that we swap hu and hv
            const float3 Qp = make_float3(Q[0][l+1][k],
                                          Q[2][l+1][k],
                                          Q[1][l+1][k]);
            const float3 Qm = make_float3(Q[0][l][k],
                                          Q[2][l][k],
                                          Q[1][l][k]);

            // Computed flux
            // Note that we swap back
            const float3 flux = LxF_2D_flux(Qm, Qp, g_, dy_, dt_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }  
}


extern "C" {
__global__ 
void LxFKernel(
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
    
    __shared__ float Q[vars][h+2][w+2];
    __shared__ float F[vars][h  ][w+1];
    __shared__ float G[vars][h+1][w  ];
    
    //Read from global memory
    readBlock<w, h, gc_x, gc_y,  1,  1>( h0_ptr_,  h0_pitch_, Q[0], nx_, ny_, boundary_conditions_);
    readBlock<w, h, gc_x, gc_y, -1,  1>(hu0_ptr_, hu0_pitch_, Q[1], nx_, ny_, boundary_conditions_);
    readBlock<w, h, gc_x, gc_y,  1, -1>(hv0_ptr_, hv0_pitch_, Q[2], nx_, ny_, boundary_conditions_);
    
    //Compute fluxes along the x and y axis
    computeFluxF<w, h>(Q, F, g_, dx_, dt_);
    computeFluxG<w, h>(Q, G, g_, dy_, dt_);
    __syncthreads();

    //Evolve for all cells
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i = tx + 1; //Skip local ghost cells, i.e., +1
    const int j = ty + 1;
    
    Q[0][j][i] += (F[0][ty][tx] - F[0][ty  ][tx+1]) * dt_ / dx_ 
                + (G[0][ty][tx] - G[0][ty+1][tx  ]) * dt_ / dy_;
    Q[1][j][i] += (F[1][ty][tx] - F[1][ty  ][tx+1]) * dt_ / dx_ 
                + (G[1][ty][tx] - G[1][ty+1][tx  ]) * dt_ / dy_;
    Q[2][j][i] += (F[2][ty][tx] - F[2][ty  ][tx+1]) * dt_ / dx_ 
                + (G[2][ty][tx] - G[2][ty+1][tx  ]) * dt_ / dy_;
    __syncthreads();

    //Write to main memory
    writeBlock<w, h, gc_x, gc_y>( h1_ptr_,  h1_pitch_, Q[0], nx_, ny_, 0, 1);
    writeBlock<w, h, gc_x, gc_y>(hu1_ptr_, hu1_pitch_, Q[1], nx_, ny_, 0, 1);
    writeBlock<w, h, gc_x, gc_y>(hv1_ptr_, hv1_pitch_, Q[2], nx_, ny_, 0, 1);
    
    //Compute the CFL for this block
    if (cfl_ != NULL) {
        writeCfl<w, h, gc_x, gc_y, vars>(Q, Q[0], nx_, ny_, dx_, dy_, g_, cfl_);
    }
}

} // extern "C"


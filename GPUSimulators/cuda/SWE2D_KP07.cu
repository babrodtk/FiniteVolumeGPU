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



#include "common.h"
#include "SWECommon.h"
#include "limiters.h"


__device__
void computeFluxF(float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  float Qx[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
                  float F[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
                  const float g_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    {
        int j=ty;
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<BLOCK_WIDTH+1; i+=BLOCK_WIDTH) {
            const int k = i + 1;
            // Q at interface from the right and left
            const float3 Qp = make_float3(Q[0][l][k+1] - 0.5f*Qx[0][j][i+1],
                                          Q[1][l][k+1] - 0.5f*Qx[1][j][i+1],
                                          Q[2][l][k+1] - 0.5f*Qx[2][j][i+1]);
            const float3 Qm = make_float3(Q[0][l][k  ] + 0.5f*Qx[0][j][i  ],
                                          Q[1][l][k  ] + 0.5f*Qx[1][j][i  ],
                                          Q[2][l][k  ] + 0.5f*Qx[2][j][i  ]);
                                       
            // Computed flux
            const float3 flux = CentralUpwindFlux(Qm, Qp, g_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }    
}

__device__
void computeFluxG(float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  float Qy[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
                  float G[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
                  const float g_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    for (int j=ty; j<BLOCK_HEIGHT+1; j+=BLOCK_HEIGHT) {
        const int l = j + 1;
        {
            int i=tx;
            const int k = i + 2; //Skip ghost cells
            // Q at interface from the right and left
            // Note that we swap hu and hv
            const float3 Qp = make_float3(Q[0][l+1][k] - 0.5f*Qy[0][j+1][i],
                                          Q[2][l+1][k] - 0.5f*Qy[2][j+1][i],
                                          Q[1][l+1][k] - 0.5f*Qy[1][j+1][i]);
            const float3 Qm = make_float3(Q[0][l  ][k] + 0.5f*Qy[0][j  ][i],
                                          Q[2][l  ][k] + 0.5f*Qy[2][j  ][i],
                                          Q[1][l  ][k] + 0.5f*Qy[1][j  ][i]);
                                       
            // Computed flux
            // Note that we swap back
            const float3 flux = CentralUpwindFlux(Qm, Qp, g_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
}




/**
  * This unsplit kernel computes the 2D numerical scheme with a TVD RK2 time integration scheme
  */
extern "C" {
__global__ void KP07Kernel(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
        float theta_,
        
        int step_,
        
        //Input h^n
        float* h0_ptr_, int h0_pitch_,
        float* hu0_ptr_, int hu0_pitch_,
        float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        float* h1_ptr_, int h1_pitch_,
        float* hu1_ptr_, int hu1_pitch_,
        float* hv1_ptr_, int hv1_pitch_) {
            
    const unsigned int w = BLOCK_WIDTH;
    const unsigned int h = BLOCK_HEIGHT;
    const unsigned int gc = 2;
    const unsigned int vars = 3;
        
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockDim.y*blockIdx.y + threadIdx.y + 2;
    
    //Shared memory variables
    __shared__ float Q[3][h+4][w+4];
    
    //The following slightly wastes memory, but enables us to reuse the 
    //funcitons in common.opencl
    __shared__ float Qx[3][h+2][w+2];
    __shared__ float Qy[3][h+2][w+2];
    __shared__ float  F[3][h+1][w+1];
    __shared__ float  G[3][h+1][w+1];
    
    
    
    //Read into shared memory
    readBlock<w, h, gc>( h0_ptr_,  h0_pitch_, Q[0], nx_+2, ny_+2);
    readBlock<w, h, gc>(hu0_ptr_, hu0_pitch_, Q[1], nx_+2, ny_+2);
    readBlock<w, h, gc>(hv0_ptr_, hv0_pitch_, Q[2], nx_+2, ny_+2);
    __syncthreads();
    
    
    //Fix boundary conditions
    noFlowBoundary<w, h, gc,  1,  1>(Q[0], nx_, ny_);
    noFlowBoundary<w, h, gc, -1,  1>(Q[1], nx_, ny_);
    noFlowBoundary<w, h, gc,  1, -1>(Q[2], nx_, ny_);
    __syncthreads();
    
    
    //Reconstruct slopes along x and axis
    minmodSlopeX(Q, Qx, theta_);
    minmodSlopeY(Q, Qy, theta_);
    __syncthreads();
    
    
    //Compute fluxes along the x and y axis
    computeFluxF(Q, Qx, F, g_);
    computeFluxG(Q, Qy, G, g_);
    __syncthreads();
    
    
    //Sum fluxes and advance in time for all internal cells
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;
        
        const float h1  = Q[0][j][i] + (F[0][ty][tx] - F[0][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[0][ty][tx] - G[0][ty+1][tx  ]) * dt_ / dy_;
        const float hu1 = Q[1][j][i] + (F[1][ty][tx] - F[1][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[1][ty][tx] - G[1][ty+1][tx  ]) * dt_ / dy_;
        const float hv1 = Q[2][j][i] + (F[2][ty][tx] - F[2][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[2][ty][tx] - G[2][ty+1][tx  ]) * dt_ / dy_;

        float* const h_row  = (float*) ((char*) h1_ptr_ + h1_pitch_*tj);
        float* const hu_row = (float*) ((char*) hu1_ptr_ + hu1_pitch_*tj);
        float* const hv_row = (float*) ((char*) hv1_ptr_ + hv1_pitch_*tj);
                    
        if  (step_ == 0) {
            //First step of RK2 ODE integrator
            
            h_row[ti] = h1;
            hu_row[ti] = hu1;
            hv_row[ti] = hv1;
        }
        else if (step_ == 1) {
            //Second step of RK2 ODE integrator
            
            //First read Q^n
            const float h_a  = h_row[ti];
            const float hu_a = hu_row[ti];
            const float hv_a = hv_row[ti];
            
            //Compute Q^n+1
            const float h_b  = 0.5f*(h_a + h1);
            const float hu_b = 0.5f*(hu_a + hu1);
            const float hv_b = 0.5f*(hv_a + hv1);
            
            //Write to main memory
            h_row[ti] = h_b;
            hu_row[ti] = hu_b;
            hv_row[ti] = hv_b;
        }
    }
}
} //extern "C"
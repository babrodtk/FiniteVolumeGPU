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



/**
  * Computes the flux along the x axis for all faces
  */
__device__
void computeFluxF(float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  float F[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
                  const float g_, const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
                      
    {
        int j=ty; 
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<BLOCK_WIDTH+1; i+=BLOCK_WIDTH) {
            const int k = i + 1;
            
            // Q at interface from the right and left
            const float3 Ql2 = make_float3(Q[0][l][k-1], Q[1][l][k-1], Q[2][l][k-1]);
            const float3 Ql1 = make_float3(Q[0][l][k  ], Q[1][l][k  ], Q[2][l][k  ]);
            const float3 Qr1 = make_float3(Q[0][l][k+1], Q[1][l][k+1], Q[2][l][k+1]);
            const float3 Qr2 = make_float3(Q[0][l][k+2], Q[1][l][k+2], Q[2][l][k+2]);

            // Computed flux
            const float3 flux = WAF_1D_flux(Ql2, Ql1, Qr1, Qr2, g_, dx_, dt_);
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
void computeFluxG(float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  float G[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
                  const float g_, const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Compute fluxes along the y axis
    for (int j=ty; j<BLOCK_HEIGHT+1; j+=BLOCK_HEIGHT) {
        const int l = j + 1;
        {
            int i=tx;
            const int k = i + 2; //Skip ghost cells
            // Q at interface from the right and left
            // Note that we swap hu and hv
            const float3 Ql2 = make_float3(Q[0][l-1][k], Q[2][l-1][k], Q[1][l-1][k]);
            const float3 Ql1 = make_float3(Q[0][l  ][k], Q[2][l  ][k], Q[1][l  ][k]);
            const float3 Qr1 = make_float3(Q[0][l+1][k], Q[2][l+1][k], Q[1][l+1][k]);
            const float3 Qr2 = make_float3(Q[0][l+2][k], Q[2][l+2][k], Q[1][l+2][k]);
            
            // Computed flux
            // Note that we swap back
            const float3 flux = WAF_1D_flux(Ql2, Ql1, Qr1, Qr2, g_, dy_, dt_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
}














extern "C" {
__global__ void WAFKernel(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_, int step_,
        
        //Input h^n
        float* h0_ptr_, int h0_pitch_,
        float* hu0_ptr_, int hu0_pitch_,
        float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        float* h1_ptr_, int h1_pitch_,
        float* hu1_ptr_, int hu1_pitch_,
        float* hv1_ptr_, int hv1_pitch_) {    
    //Shared memory variables
    __shared__ float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4];
    __shared__ float F[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1];
    
    
    
    //Read into shared memory Q from global memory
    float* Q_ptr[3] = {h0_ptr_, hu0_ptr_, hv0_ptr_};
    int Q_pitch[3] = {h0_pitch_, hu0_pitch_, hv0_pitch_};
    readBlock<3, BLOCK_WIDTH+4, BLOCK_HEIGHT+4, BLOCK_WIDTH, BLOCK_HEIGHT>(Q_ptr, Q_pitch, Q, nx_+4, ny_+4);
    __syncthreads();
    
    
    //Set boundary conditions
    noFlowBoundary2(Q, nx_, ny_);
    __syncthreads();
    
    
    
    //Step 0 => evolve x first, then y
    if (step_ == 0) {
        //Compute fluxes along the x axis and evolve
        computeFluxF(Q, F, g_, dx_, dt_);
        __syncthreads();
        evolveF2(Q, F, nx_, ny_, dx_, dt_);
        __syncthreads();
        
        //Fix boundary conditions
        noFlowBoundary2(Q, nx_, ny_);
        __syncthreads();
        
        //Compute fluxes along the y axis and evolve
        computeFluxG(Q, F, g_, dy_, dt_);
        __syncthreads();
        evolveG2(Q, F, nx_, ny_, dy_, dt_);
        __syncthreads();
    }
    //Step 1 => evolve y first, then x
    else {
        //Compute fluxes along the y axis and evolve
        computeFluxG(Q, F, g_, dy_, dt_);
        __syncthreads();
        evolveG2(Q, F, nx_, ny_, dy_, dt_);
        __syncthreads();
        
        //Fix boundary conditions
        noFlowBoundary2(Q, nx_, ny_);
        __syncthreads();
        
        //Compute fluxes along the x axis and evolve
        computeFluxF(Q, F, g_, dx_, dt_);
        __syncthreads();
        evolveF2(Q, F, nx_, ny_, dx_, dt_);
        __syncthreads();
    }


    
    // Write to main memory for all internal cells
    writeBlock2(h1_ptr_, h1_pitch_,
                hu1_ptr_, hu1_pitch_,
                hv1_ptr_, hv1_pitch_,
                Q, nx_, ny_);
}

} // extern "C"
/*
This GPU kernel implements the HLL flux

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
                  const float g_) {
    for (int j=threadIdx.y; j<BLOCK_HEIGHT+2; j+=BLOCK_HEIGHT) {
        for (int i=threadIdx.x; i<BLOCK_WIDTH+1; i+=BLOCK_WIDTH) {
            // Q at interface from the right and left
            const float3 Q_r = make_float3(Q[0][j][i+1],
                                           Q[1][j][i+1],
                                           Q[2][j][i+1]);
            const float3 Q_l = make_float3(Q[0][j][i],
                                           Q[1][j][i],
                                           Q[2][j][i]);
                       
            // Computed flux
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
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
                  const float g_) {
    //Compute fluxes along the y axis
    for (int j=threadIdx.y; j<BLOCK_HEIGHT+1; j+=BLOCK_HEIGHT) {
        for (int i=threadIdx.x; i<BLOCK_WIDTH+2; i+=BLOCK_WIDTH) {
            // Q at interface from the right and left
            // Note that we swap hu and hv
            const float3 Q_r = make_float3(Q[0][j+1][i],
                                           Q[2][j+1][i],
                                           Q[1][j+1][i]);
            const float3 Q_l = make_float3(Q[0][j][i],
                                           Q[2][j][i],
                                           Q[1][j][i]);
                                       
            // Computed flux
            //Note that we here swap hu and hv back to the original
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
}












extern "C" {
    
__global__ void HLLKernel(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
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
    const unsigned int gc = 1;
    const unsigned int vars = 3;
    
    //Shared memory variables
    __shared__ float Q[3][h+2][w+2];
    __shared__ float F[3][h+2][w+2];
    
    //Read into shared memory
    readBlock<w, h, gc>( h0_ptr_,  h0_pitch_, Q[0], nx_+2, ny_+2);
    readBlock<w, h, gc>(hu0_ptr_, hu0_pitch_, Q[1], nx_+2, ny_+2);
    readBlock<w, h, gc>(hv0_ptr_, hv0_pitch_, Q[2], nx_+2, ny_+2);
    __syncthreads();

    //Set boundary conditions
    noFlowBoundary<w, h, gc,  1,  1>(Q[0], nx_, ny_);
    noFlowBoundary<w, h, gc, -1,  1>(Q[1], nx_, ny_);
    noFlowBoundary<w, h, gc,  1, -1>(Q[2], nx_, ny_);
    __syncthreads();
    
    //Compute F flux
    computeFluxF(Q, F, g_);
    __syncthreads();
    
    evolveF<w, h, gc, vars>(Q, F, dx_, dt_);
    __syncthreads();
    
    //Set boundary conditions
    noFlowBoundary<w, h, gc,  1,  1>(Q[0], nx_, ny_);
    noFlowBoundary<w, h, gc, -1,  1>(Q[1], nx_, ny_);
    noFlowBoundary<w, h, gc,  1, -1>(Q[2], nx_, ny_);
    __syncthreads();
    
    //Compute G flux
    computeFluxG(Q, F, g_);
    __syncthreads();
    
    evolveG<w, h, gc, vars>(Q, F, dy_, dt_);
    __syncthreads();
    
    // Write to main memory for all internal cells
    writeBlock<w, h, gc>( h1_ptr_,  h1_pitch_, Q[0], nx_, ny_);
    writeBlock<w, h, gc>(hu1_ptr_, hu1_pitch_, Q[1], nx_, ny_);
    writeBlock<w, h, gc>(hv1_ptr_, hv1_pitch_, Q[2], nx_, ny_);
}

} // extern "C"
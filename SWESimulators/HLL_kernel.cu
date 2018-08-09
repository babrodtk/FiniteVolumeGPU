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



#include "common.cu"
#include "fluxes/HartenLaxVanLeer.cu"





/**
  * Computes the flux along the x axis for all faces
  */
__device__
void computeFluxF(float Q[3][block_height+2][block_width+2],
                  float F[3][block_height+1][block_width+1],
                  const float g_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    {
        const int j=ty;
        const int l = j + 1; //Skip ghost cells     
        for (int i=tx; i<block_width+1; i+=block_width) { 
            const int k = i;
            
            const float3 Q_l  = make_float3(Q[0][l][k  ], Q[1][l][k  ], Q[2][l][k  ]);
            const float3 Q_r  = make_float3(Q[0][l][k+1], Q[1][l][k+1], Q[2][l][k+1]);
            
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
            
            //Write to shared memory
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
void computeFluxG(float Q[3][block_height+2][block_width+2],
                  float G[3][block_height+1][block_width+1],
                  const float g_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height+1; j+=block_height) {
        const int l = j;
        {
            const int i=tx;
            const int k = i + 1; //Skip ghost cells
            
            //NOte that hu and hv are swapped ("transposing" the domain)!
            const float3 Q_l = make_float3(Q[0][l  ][k], Q[2][l  ][k], Q[1][l  ][k]);
            const float3 Q_r = make_float3(Q[0][l+1][k], Q[2][l+1][k], Q[1][l+1][k]);
                                       
            // Computed flux
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
            
            //Write to shared memory
            //Note that we here swap hu and hv back to the original
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
}













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
    //Shared memory variables
    __shared__ float Q[3][block_height+2][block_width+2];
    __shared__ float F[3][block_height+1][block_width+1];
    
    
    //Read into shared memory
    readBlock1(h0_ptr_, h0_pitch_,
               hu0_ptr_, hu0_pitch_,
               hv0_ptr_, hv0_pitch_,
               Q, nx_, ny_);
    __syncthreads();

    noFlowBoundary1(Q, nx_, ny_);
    __syncthreads();
    
    //Compute F flux
    computeFluxF(Q, F, g_);
    __syncthreads();
    evolveF1(Q, F, nx_, ny_, dx_, dt_);
    __syncthreads();
    
    //Set boundary conditions
    noFlowBoundary1(Q, nx_, ny_);
    __syncthreads();
    
    //Compute G flux
    computeFluxG(Q, F, g_);
    __syncthreads();
    evolveG1(Q, F, nx_, ny_, dy_, dt_);
    __syncthreads();
    
    
    //Q[0][get_local_id(1) + 1][get_local_id(0) + 1] += 0.1;
    
    
    
    // Write to main memory for all internal cells
    writeBlock1(h1_ptr_, h1_pitch_,
                hu1_ptr_, hu1_pitch_,
                hv1_ptr_, hv1_pitch_,
                Q, nx_, ny_);
}
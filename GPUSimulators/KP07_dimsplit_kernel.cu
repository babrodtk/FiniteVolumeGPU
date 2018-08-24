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



#include "common.cu"
#include "SWECommon.cu"
#include "limiters.cu"
#include "fluxes/CentralUpwind.cu"


__device__
void computeFluxF(float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  float Qx[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
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
            // Reconstruct point values of Q at the left and right hand side 
            // of the cell for both the left (i) and right (i+1) cell 
            const float3 Q_rl = make_float3(Q[0][l][k+1] - 0.5f*Qx[0][j][i+1],
                                            Q[1][l][k+1] - 0.5f*Qx[1][j][i+1],
                                            Q[2][l][k+1] - 0.5f*Qx[2][j][i+1]);
            const float3 Q_rr = make_float3(Q[0][l][k+1] + 0.5f*Qx[0][j][i+1],
                                            Q[1][l][k+1] + 0.5f*Qx[1][j][i+1],
                                            Q[2][l][k+1] + 0.5f*Qx[2][j][i+1]);
                                         
            const float3 Q_ll = make_float3(Q[0][l][k] - 0.5f*Qx[0][j][i],
                                            Q[1][l][k] - 0.5f*Qx[1][j][i],
                                            Q[2][l][k] - 0.5f*Qx[2][j][i]);
            const float3 Q_lr = make_float3(Q[0][l][k] + 0.5f*Qx[0][j][i],
                                            Q[1][l][k] + 0.5f*Qx[1][j][i],
                                            Q[2][l][k] + 0.5f*Qx[2][j][i]);
                                    
            //Evolve half a timestep (predictor step)
            const float3 Q_r_bar = Q_rl + dt_/(2.0f*dx_) * (F_func(Q_rl, g_) - F_func(Q_rr, g_));
            const float3 Q_l_bar = Q_lr + dt_/(2.0f*dx_) * (F_func(Q_ll, g_) - F_func(Q_lr, g_));

            // Compute flux based on prediction
            const float3 flux = CentralUpwindFlux(Q_l_bar, Q_r_bar, g_);
            
            //Write to shared memory
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
                  const float g_, const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    for (int j=ty; j<BLOCK_HEIGHT+1; j+=BLOCK_HEIGHT) {
        const int l = j + 1;
        {
            int i=tx;
            const int k = i + 2; //Skip ghost cells
            // Reconstruct point values of Q at the left and right hand side 
            // of the cell for both the left (i) and right (i+1) cell 
            //NOte that hu and hv are swapped ("transposing" the domain)!
            const float3 Q_rl = make_float3(Q[0][l+1][k] - 0.5f*Qy[0][j+1][i],
                                            Q[2][l+1][k] - 0.5f*Qy[2][j+1][i],
                                            Q[1][l+1][k] - 0.5f*Qy[1][j+1][i]);
            const float3 Q_rr = make_float3(Q[0][l+1][k] + 0.5f*Qy[0][j+1][i],
                                            Q[2][l+1][k] + 0.5f*Qy[2][j+1][i],
                                            Q[1][l+1][k] + 0.5f*Qy[1][j+1][i]);
                                        
            const float3 Q_ll = make_float3(Q[0][l][k] - 0.5f*Qy[0][j][i],
                                            Q[2][l][k] - 0.5f*Qy[2][j][i],
                                            Q[1][l][k] - 0.5f*Qy[1][j][i]);
            const float3 Q_lr = make_float3(Q[0][l][k] + 0.5f*Qy[0][j][i],
                                            Q[2][l][k] + 0.5f*Qy[2][j][i],
                                            Q[1][l][k] + 0.5f*Qy[1][j][i]);
                                     
            //Evolve half a timestep (predictor step)
            const float3 Q_r_bar = Q_rl + dt_/(2.0f*dy_) * (F_func(Q_rl, g_) - F_func(Q_rr, g_));
            const float3 Q_l_bar = Q_lr + dt_/(2.0f*dy_) * (F_func(Q_ll, g_) - F_func(Q_lr, g_));
            
            // Compute flux based on prediction
            const float3 flux = CentralUpwindFlux(Q_l_bar, Q_r_bar, g_);
            
            //Write to shared memory
            //Note that we here swap hu and hv back to the original
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
__global__ void KP07DimsplitKernel(
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
        
        
    //Shared memory variables
    __shared__ float Q[3][BLOCK_HEIGHT+4][BLOCK_WIDTH+4];
    __shared__ float Qx[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2];
    __shared__ float F[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1];
    
    
    
    //Read into shared memory
    readBlock<BLOCK_WIDTH+4, BLOCK_HEIGHT+4>(h0_ptr_, h0_pitch_, Q[0], nx_+3, ny_+3);
    readBlock<BLOCK_WIDTH+4, BLOCK_HEIGHT+4>(hu0_ptr_, hu0_pitch_, Q[1], nx_+3, ny_+3);
    readBlock<BLOCK_WIDTH+4, BLOCK_HEIGHT+4>(hv0_ptr_, hv0_pitch_, Q[2], nx_+3, ny_+3);
    __syncthreads();
    
    
    //Fix boundary conditions
    noFlowBoundary2(Q, nx_, ny_);
    __syncthreads();
    
    
    
    //Step 0 => evolve x first, then y
    if (step_ == 0) {
        //Compute fluxes along the x axis and evolve
        minmodSlopeX(Q, Qx, theta_);
        __syncthreads();
        computeFluxF(Q, Qx, F, g_, dx_, dt_);
        __syncthreads();
        evolveF2(Q, F, nx_, ny_, dx_, dt_);
        __syncthreads();
        
        //Set boundary conditions
        noFlowBoundary2(Q, nx_, ny_);
        __syncthreads();
        
        //Compute fluxes along the y axis and evolve
        minmodSlopeY(Q, Qx, theta_);
        __syncthreads();
        computeFluxG(Q, Qx, F, g_, dy_, dt_);
        __syncthreads();
        evolveG2(Q, F, nx_, ny_, dy_, dt_);
        __syncthreads();
    }
    //Step 1 => evolve y first, then x
    else {
        //Compute fluxes along the y axis and evolve
        minmodSlopeY(Q, Qx, theta_);
        __syncthreads();
        computeFluxG(Q, Qx, F, g_, dy_, dt_);
        __syncthreads();
        evolveG2(Q, F, nx_, ny_, dy_, dt_);
        __syncthreads();
        
        //Set boundary conditions
        noFlowBoundary2(Q, nx_, ny_);
        __syncthreads();
        
        //Compute fluxes along the x axis and evolve
        minmodSlopeX(Q, Qx, theta_);
        __syncthreads();
        computeFluxF(Q, Qx, F, g_, dx_, dt_);
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
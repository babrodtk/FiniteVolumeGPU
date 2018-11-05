 /*
This kernel implements the Central Upwind flux function to
solve the Euler equations 

Copyright (C) 2018  SINTEF Digital

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
#include "EulerCommon.h"
#include "limiters.h"


__device__
void computeFluxF(float Q[4][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  float Qx[4][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  float F[4][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  const float gamma_, const float dx_, const float dt_) {
    for (int j=threadIdx.y; j<BLOCK_HEIGHT+4; j+=BLOCK_HEIGHT) {
        for (int i=threadIdx.x+1; i<BLOCK_WIDTH+2; i+=BLOCK_WIDTH) {
            // Reconstruct point values of Q at the left and right hand side 
            // of the cell for both the left (i) and right (i+1) cell 
            const float4 Q_rl = make_float4(Q[0][j][i+1] - 0.5f*Qx[0][j][i+1],
                                            Q[1][j][i+1] - 0.5f*Qx[1][j][i+1],
                                            Q[2][j][i+1] - 0.5f*Qx[2][j][i+1],
                                            Q[3][j][i+1] - 0.5f*Qx[3][j][i+1]);
            const float4 Q_rr = make_float4(Q[0][j][i+1] + 0.5f*Qx[0][j][i+1],
                                            Q[1][j][i+1] + 0.5f*Qx[1][j][i+1],
                                            Q[2][j][i+1] + 0.5f*Qx[2][j][i+1],
                                            Q[3][j][i+1] + 0.5f*Qx[3][j][i+1]);

            const float4 Q_ll = make_float4(Q[0][j][i] - 0.5f*Qx[0][j][i],
                                            Q[1][j][i] - 0.5f*Qx[1][j][i],
                                            Q[2][j][i] - 0.5f*Qx[2][j][i],
                                            Q[3][j][i] - 0.5f*Qx[3][j][i]);
            const float4 Q_lr = make_float4(Q[0][j][i] + 0.5f*Qx[0][j][i],
                                            Q[1][j][i] + 0.5f*Qx[1][j][i],
                                            Q[2][j][i] + 0.5f*Qx[2][j][i],
                                            Q[3][j][i] + 0.5f*Qx[3][j][i]);


            //Evolve half a timestep (predictor step)
            const float4 Q_r_bar = Q_rl + dt_/(2.0f*dx_) * (F_func(Q_rl, gamma_) - F_func(Q_rr, gamma_));
            const float4 Q_l_bar = Q_lr + dt_/(2.0f*dx_) * (F_func(Q_ll, gamma_) - F_func(Q_lr, gamma_));

            // Compute flux based on prediction
            const float4 flux = CentralUpwindFlux(Q_l_bar, Q_r_bar, gamma_);
            
            //Write to shared memory
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
            F[3][j][i] = flux.w;
        }
    }
}

__device__
void computeFluxG(float Q[4][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  float Qy[4][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  float G[4][BLOCK_HEIGHT+4][BLOCK_WIDTH+4],
                  const float gamma_, const float dy_, const float dt_) {
    for (int j=threadIdx.y+1; j<BLOCK_HEIGHT+2; j+=BLOCK_HEIGHT) {
        for (int i=threadIdx.x; i<BLOCK_WIDTH+4; i+=BLOCK_WIDTH) {
            // Reconstruct point values of Q at the left and right hand side 
            // of the cell for both the left (i) and right (i+1) cell 
            //NOte that hu and hv are swapped ("transposing" the domain)!
            const float4 Q_rl = make_float4(Q[0][j+1][i] - 0.5f*Qy[0][j+1][i],
                                            Q[2][j+1][i] - 0.5f*Qy[2][j+1][i],
                                            Q[1][j+1][i] - 0.5f*Qy[1][j+1][i],
                                            Q[3][j+1][i] - 0.5f*Qy[3][j+1][i]);
            const float4 Q_rr = make_float4(Q[0][j+1][i] + 0.5f*Qy[0][j+1][i],
                                            Q[2][j+1][i] + 0.5f*Qy[2][j+1][i],
                                            Q[1][j+1][i] + 0.5f*Qy[1][j+1][i],
                                            Q[3][j+1][i] + 0.5f*Qy[3][j+1][i]);

            const float4 Q_ll = make_float4(Q[0][j][i] - 0.5f*Qy[0][j][i],
                                            Q[2][j][i] - 0.5f*Qy[2][j][i],
                                            Q[1][j][i] - 0.5f*Qy[1][j][i],
                                            Q[3][j][i] - 0.5f*Qy[3][j][i]);
            const float4 Q_lr = make_float4(Q[0][j][i] + 0.5f*Qy[0][j][i],
                                            Q[2][j][i] + 0.5f*Qy[2][j][i],
                                            Q[1][j][i] + 0.5f*Qy[1][j][i],
                                            Q[3][j][i] + 0.5f*Qy[3][j][i]);

            //Evolve half a timestep (predictor step)
            const float4 Q_r_bar = Q_rl + dt_/(2.0f*dy_) * (F_func(Q_rl, gamma_) - F_func(Q_rr, gamma_));
            const float4 Q_l_bar = Q_lr + dt_/(2.0f*dy_) * (F_func(Q_ll, gamma_) - F_func(Q_lr, gamma_));
            
            // Compute flux based on prediction
            const float4 flux = CentralUpwindFlux(Q_l_bar, Q_r_bar, gamma_);
            
            //Write to shared memory
            //Note that we here swap hu and hv back to the original
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
            G[3][j][i] = flux.w;
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
        float gamma_,
        
        float theta_,
        
        int step_,
        
        //Input h^n
        float* rho0_ptr_, int rho0_pitch_,
        float* rho_u0_ptr_, int rho_u0_pitch_,
        float* rho_v0_ptr_, int rho_v0_pitch_,
        float* E0_ptr_, int E0_pitch_,
        
        //Output h^{n+1}
        float* rho1_ptr_, int rho1_pitch_,
        float* rho_u1_ptr_, int rho_u1_pitch_,
        float* rho_v1_ptr_, int rho_v1_pitch_,
        float* E1_ptr_, int E1_pitch_) {
        
    const unsigned int w = BLOCK_WIDTH;
    const unsigned int h = BLOCK_HEIGHT;
    const unsigned int gc = 2;
    const unsigned int vars = 4;
        
    //Shared memory variables
    __shared__ float  Q[4][h+4][w+4];
    __shared__ float Qx[4][h+4][w+4];
    __shared__ float  F[4][h+4][w+4];
    
    
    
    //Read into shared memory
    readBlock<w, h, gc>(  rho0_ptr_,   rho0_pitch_, Q[0], nx_, ny_);
    readBlock<w, h, gc>(rho_u0_ptr_, rho_u0_pitch_, Q[1], nx_, ny_);
    readBlock<w, h, gc>(rho_v0_ptr_, rho_v0_pitch_, Q[2], nx_, ny_);
    readBlock<w, h, gc>(    E0_ptr_,     E0_pitch_, Q[3], nx_, ny_);
    __syncthreads();
    
    //Fix boundary conditions
    noFlowBoundary<w, h, gc,  1,  1>(Q[0], nx_, ny_);
    noFlowBoundary<w, h, gc,  1,  1>(Q[1], nx_, ny_);
    noFlowBoundary<w, h, gc,  1, -1>(Q[2], nx_, ny_);
    noFlowBoundary<w, h, gc,  1,  1>(Q[3], nx_, ny_);
    __syncthreads();


    //Step 0 => evolve x first, then y
    if (step_ == 0) {
        //Compute fluxes along the x axis and evolve
        minmodSlopeX<w, h, gc, vars>(Q, Qx, theta_);
        __syncthreads();

        computeFluxF(Q, Qx, F, gamma_, dx_, dt_);
        __syncthreads();

        evolveF<w, h, gc, vars>(Q, F, dx_, dt_);
        __syncthreads();

        //Set boundary conditions
        noFlowBoundary<w, h, gc,  1,  1>(Q[0], nx_, ny_);
        noFlowBoundary<w, h, gc,  1,  1>(Q[1], nx_, ny_);
        noFlowBoundary<w, h, gc,  1, -1>(Q[2], nx_, ny_);
        noFlowBoundary<w, h, gc,  1,  1>(Q[3], nx_, ny_);
        __syncthreads();

        //Compute fluxes along the y axis and evolve
        minmodSlopeY<w, h, gc, vars>(Q, Qx, theta_);
        __syncthreads();

        computeFluxG(Q, Qx, F, gamma_, dy_, dt_);
        __syncthreads();

        evolveG<w, h, gc, vars>(Q, F, dy_, dt_);
        __syncthreads();    

    }
    //Step 1 => evolve y first, then x
    else {
        //Compute fluxes along the y axis and evolve
        minmodSlopeY<w, h, gc, vars>(Q, Qx, theta_);
        __syncthreads();
  
        computeFluxG(Q, Qx, F, gamma_, dy_, dt_);
        __syncthreads();
  
        evolveG<w, h, gc, vars>(Q, F, dy_, dt_);
        __syncthreads();
  
        //Set boundary conditions
        noFlowBoundary<w, h, gc,  1,  1>(Q[0], nx_, ny_);
        noFlowBoundary<w, h, gc,  1,  1>(Q[1], nx_, ny_);
        noFlowBoundary<w, h, gc,  1, -1>(Q[2], nx_, ny_);
        noFlowBoundary<w, h, gc,  1,  1>(Q[3], nx_, ny_);
        __syncthreads();
        
        //Compute fluxes along the x axis and evolve
        minmodSlopeX<w, h, gc, vars>(Q, Qx, theta_);
        __syncthreads();

        computeFluxF(Q, Qx, F, gamma_, dx_, dt_);
        __syncthreads();

        evolveF<w, h, gc, vars>(Q, F, dx_, dt_);
        __syncthreads();
        
        //This is the RK2-part
        const int tx = threadIdx.x + gc;
        const int ty = threadIdx.y + gc;
        const float q1 = Q[0][ty][tx];
        const float q2 = Q[1][ty][tx];
        const float q3 = Q[2][ty][tx];
        const float q4 = Q[3][ty][tx];
        __syncthreads();
        
        readBlock<w, h, gc>(  rho1_ptr_,   rho1_pitch_, Q[0], nx_, ny_);
        readBlock<w, h, gc>(rho_u1_ptr_, rho_u1_pitch_, Q[1], nx_, ny_);
        readBlock<w, h, gc>(rho_v1_ptr_, rho_v1_pitch_, Q[2], nx_, ny_);
        readBlock<w, h, gc>(    E1_ptr_,     E1_pitch_, Q[3], nx_, ny_);
        __syncthreads();
        
        Q[0][ty][tx] = 0.5f*( Q[0][ty][tx] + q1 );
        Q[1][ty][tx] = 0.5f*( Q[1][ty][tx] + q2 );
        Q[2][ty][tx] = 0.5f*( Q[2][ty][tx] + q3 );
        Q[3][ty][tx] = 0.5f*( Q[3][ty][tx] + q4 );
    }

    
    // Write to main memory for all internal cells
    writeBlock<w, h, gc>(  rho1_ptr_,   rho1_pitch_, Q[0], nx_, ny_);
    writeBlock<w, h, gc>(rho_u1_ptr_, rho_u1_pitch_, Q[1], nx_, ny_);
    writeBlock<w, h, gc>(rho_v1_ptr_, rho_v1_pitch_, Q[2], nx_, ny_);
    writeBlock<w, h, gc>(    E1_ptr_,     E1_pitch_, Q[3], nx_, ny_);
}

} // extern "C"
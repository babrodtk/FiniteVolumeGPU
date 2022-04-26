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
            //const float4 flux = CentralUpwindFlux(Q_l_bar, Q_r_bar, gamma_);
            const float4 flux = HLL_flux(Q_l_bar, Q_r_bar, gamma_);
            
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
            //const float4 flux = HLL_flux(Q_l_bar, Q_r_bar, gamma_);
            
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
        float g_,
        float gamma_,
        
        float theta_,
        
        int step_,
        int boundary_conditions_,
        
        //Input h^n
        float* rho0_ptr_, int rho0_pitch_,
        float* rho_u0_ptr_, int rho_u0_pitch_,
        float* rho_v0_ptr_, int rho_v0_pitch_,
        float* E0_ptr_, int E0_pitch_,
        
        //Output h^{n+1}
        float* rho1_ptr_, int rho1_pitch_,
        float* rho_u1_ptr_, int rho_u1_pitch_,
        float* rho_v1_ptr_, int rho_v1_pitch_,
        float* E1_ptr_, int E1_pitch_, 
        
        //Output CFL
        float* cfl_,

        //Subarea of internal domain to compute
        int x0=0, int y0=0,
        int x1=0, int y1=0) {

    if(x1 == 0)
        x1 = nx_;

    if(y1 == 0)
        y1 = ny_;
    
    const unsigned int w = BLOCK_WIDTH;
    const unsigned int h = BLOCK_HEIGHT;
    const unsigned int gc_x = 2;
    const unsigned int gc_y = 2;
    const unsigned int vars = 4;
    
    //Shared memory variables
    __shared__ float  Q[4][h+2*gc_y][w+2*gc_x];
    __shared__ float Qx[4][h+2*gc_y][w+2*gc_x];
    __shared__ float  F[4][h+2*gc_y][w+2*gc_x];
    
    //Read into shared memory
    readBlock<w, h, gc_x, gc_y,  1,  1>(  rho0_ptr_,   rho0_pitch_, Q[0], nx_, ny_, boundary_conditions_, x0, y0, x1, y1);
    readBlock<w, h, gc_x, gc_y, -1,  1>(rho_u0_ptr_, rho_u0_pitch_, Q[1], nx_, ny_, boundary_conditions_, x0, y0, x1, y1);
    readBlock<w, h, gc_x, gc_y,  1, -1>(rho_v0_ptr_, rho_v0_pitch_, Q[2], nx_, ny_, boundary_conditions_, x0, y0, x1, y1);
    readBlock<w, h, gc_x, gc_y,  1,  1>(    E0_ptr_,     E0_pitch_, Q[3], nx_, ny_, boundary_conditions_, x0, y0, x1, y1);

    //Step 0 => evolve x first, then y
    if (step_ == 0) {
        //Compute fluxes along the x axis and evolve
        minmodSlopeX<w, h, gc_x, gc_y, vars>(Q, Qx, theta_);
        __syncthreads();
        computeFluxF(Q, Qx, F, gamma_, dx_, dt_);
        __syncthreads();
        evolveF<w, h, gc_x, gc_y, vars>(Q, F, dx_, dt_);
        __syncthreads();

        //Compute fluxes along the y axis and evolve
        minmodSlopeY<w, h, gc_x, gc_y, vars>(Q, Qx, theta_);
        __syncthreads();
        computeFluxG(Q, Qx, F, gamma_, dy_, dt_);
        __syncthreads();
        evolveG<w, h, gc_x, gc_y, vars>(Q, F, dy_, dt_);
        __syncthreads();    
        
        //Gravity source term
        if (g_ > 0.0f) {
            const int i = threadIdx.x + gc_x;
            const int j = threadIdx.y + gc_y;
            const float rho_v = Q[2][j][i];
            Q[2][j][i] -= g_*Q[0][j][i]*dt_;
            Q[3][j][i] -= g_*rho_v*dt_;
            __syncthreads();
        }
    }
    //Step 1 => evolve y first, then x
    else {
        //Compute fluxes along the y axis and evolve
        minmodSlopeY<w, h, gc_x, gc_y, vars>(Q, Qx, theta_);
        __syncthreads();
        computeFluxG(Q, Qx, F, gamma_, dy_, dt_);
        __syncthreads();
        evolveG<w, h, gc_x, gc_y, vars>(Q, F, dy_, dt_);
        __syncthreads();
        
        //Compute fluxes along the x axis and evolve
        minmodSlopeX<w, h, gc_x, gc_y, vars>(Q, Qx, theta_);
        __syncthreads();
        computeFluxF(Q, Qx, F, gamma_, dx_, dt_);
        __syncthreads();
        evolveF<w, h, gc_x, gc_y, vars>(Q, F, dx_, dt_);
        __syncthreads();
        
        //Gravity source term
        if (g_ > 0.0f) {
            const int i = threadIdx.x + gc_x;
            const int j = threadIdx.y + gc_y;
            const float rho_v = Q[2][j][i];
            Q[2][j][i] -= g_*Q[0][j][i]*dt_;
            Q[3][j][i] -= g_*rho_v*dt_;
            __syncthreads();
        }
    }

    
    // Write to main memory for all internal cells
    writeBlock<w, h, gc_x, gc_y>(  rho1_ptr_,   rho1_pitch_, Q[0], nx_, ny_, 0, 1, x0, y0, x1, y1);
    writeBlock<w, h, gc_x, gc_y>(rho_u1_ptr_, rho_u1_pitch_, Q[1], nx_, ny_, 0, 1, x0, y0, x1, y1);
    writeBlock<w, h, gc_x, gc_y>(rho_v1_ptr_, rho_v1_pitch_, Q[2], nx_, ny_, 0, 1, x0, y0, x1, y1);
    writeBlock<w, h, gc_x, gc_y>(    E1_ptr_,     E1_pitch_, Q[3], nx_, ny_, 0, 1, x0, y0, x1, y1);
    
    //Compute the CFL for this block
    if (cfl_ != NULL) {
        writeCfl<w, h, gc_x, gc_y, vars>(Q, F[0], nx_, ny_, dx_, dy_, gamma_, cfl_);
    }
}


} // extern "C"
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


template <int w, int h, int gc_x, int gc_y>
__device__
void computeFluxF(float Q[3][h+2*gc_y][w+2*gc_x],
                  float Qx[3][h+2*gc_y][w+2*gc_x],
                  float F[3][h+2*gc_y][w+2*gc_x],
                  const float g_, const float dx_, const float dt_) {
    for (int j=threadIdx.y; j<h+2*gc_y; j+=h) {
        for (int i=threadIdx.x+1; i<w+2*gc_x-2; i+=w) {
            // Reconstruct point values of Q at the left and right hand side 
            // of the cell for both the left (i) and right (i+1) cell 
            const float3 Q_rl = make_float3(Q[0][j][i+1] - 0.5f*Qx[0][j][i+1],
                                            Q[1][j][i+1] - 0.5f*Qx[1][j][i+1],
                                            Q[2][j][i+1] - 0.5f*Qx[2][j][i+1]);
            const float3 Q_rr = make_float3(Q[0][j][i+1] + 0.5f*Qx[0][j][i+1],
                                            Q[1][j][i+1] + 0.5f*Qx[1][j][i+1],
                                            Q[2][j][i+1] + 0.5f*Qx[2][j][i+1]);

            const float3 Q_ll = make_float3(Q[0][j][i] - 0.5f*Qx[0][j][i],
                                            Q[1][j][i] - 0.5f*Qx[1][j][i],
                                            Q[2][j][i] - 0.5f*Qx[2][j][i]);
            const float3 Q_lr = make_float3(Q[0][j][i] + 0.5f*Qx[0][j][i],
                                            Q[1][j][i] + 0.5f*Qx[1][j][i],
                                            Q[2][j][i] + 0.5f*Qx[2][j][i]);
                                    
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

template <int w, int h, int gc_x, int gc_y>
__device__
void computeFluxG(float Q[3][h+2*gc_y][w+2*gc_x],
                  float Qy[3][h+2*gc_y][w+2*gc_x],
                  float G[3][h+2*gc_y][w+2*gc_x],
                  const float g_, const float dy_, const float dt_) {
    for (int j=threadIdx.y+1; j<h+2*gc_y-2; j+=h) {
        for (int i=threadIdx.x; i<w+2*gc_x; i+=w) {
            // Reconstruct point values of Q at the left and right hand side 
            // of the cell for both the left (i) and right (i+1) cell 
            //NOte that hu and hv are swapped ("transposing" the domain)!
            const float3 Q_rl = make_float3(Q[0][j+1][i] - 0.5f*Qy[0][j+1][i],
                                            Q[2][j+1][i] - 0.5f*Qy[2][j+1][i],
                                            Q[1][j+1][i] - 0.5f*Qy[1][j+1][i]);
            const float3 Q_rr = make_float3(Q[0][j+1][i] + 0.5f*Qy[0][j+1][i],
                                            Q[2][j+1][i] + 0.5f*Qy[2][j+1][i],
                                            Q[1][j+1][i] + 0.5f*Qy[1][j+1][i]);
                                        
            const float3 Q_ll = make_float3(Q[0][j][i] - 0.5f*Qy[0][j][i],
                                            Q[2][j][i] - 0.5f*Qy[2][j][i],
                                            Q[1][j][i] - 0.5f*Qy[1][j][i]);
            const float3 Q_lr = make_float3(Q[0][j][i] + 0.5f*Qy[0][j][i],
                                            Q[2][j][i] + 0.5f*Qy[2][j][i],
                                            Q[1][j][i] + 0.5f*Qy[1][j][i]);
                                     
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
    const unsigned int gc_x = 2;
    const unsigned int gc_y = 2;
    const unsigned int vars = 3;
        
    //Shared memory variables
    __shared__ float  Q[vars][h+2*gc_y][w+2*gc_x];
    __shared__ float Qx[vars][h+2*gc_y][w+2*gc_x];
    __shared__ float  F[vars][h+2*gc_y][w+2*gc_x];
    
    //Read into shared memory
    readBlock<w, h, gc_x, gc_y,  1,  1>( h0_ptr_,  h0_pitch_, Q[0], nx_, ny_, boundary_conditions_);
    readBlock<w, h, gc_x, gc_y, -1,  1>(hu0_ptr_, hu0_pitch_, Q[1], nx_, ny_, boundary_conditions_);
    readBlock<w, h, gc_x, gc_y,  1, -1>(hv0_ptr_, hv0_pitch_, Q[2], nx_, ny_, boundary_conditions_);
    
    if (step_ == 0) {
        //Along X
        minmodSlopeX<w, h, gc_x, gc_y, vars>(Q, Qx, theta_);
        __syncthreads();
        computeFluxF<w, h, gc_x, gc_y>(Q, Qx, F, g_, dx_, dt_);
        __syncthreads();
        evolveF<w, h, gc_x, gc_y, vars>(Q, F, dx_, dt_);
        __syncthreads();
        
        //Along Y
        minmodSlopeY<w, h, gc_x, gc_y, vars>(Q, Qx, theta_);
        __syncthreads();
        computeFluxG<w, h, gc_x, gc_y>(Q, Qx, F, g_, dy_, dt_);
        __syncthreads();
        evolveG<w, h, gc_x, gc_y, vars>(Q, F, dy_, dt_);
        __syncthreads();
    }
    else {
        //Along Y
        minmodSlopeY<w, h, gc_x, gc_y, vars>(Q, Qx, theta_);
        __syncthreads();
        computeFluxG<w, h, gc_x, gc_y>(Q, Qx, F, g_, dy_, dt_);
        __syncthreads();
        evolveG<w, h, gc_x, gc_y, vars>(Q, F, dy_, dt_);
        __syncthreads();
        
        //Along X
        minmodSlopeX<w, h, gc_x, gc_y, vars>(Q, Qx, theta_);
        __syncthreads();
        computeFluxF<w, h, gc_x, gc_y>(Q, Qx, F, g_, dx_, dt_);
        __syncthreads();
        evolveF<w, h, gc_x, gc_y, vars>(Q, F, dx_, dt_);
        __syncthreads();
    }
    
    // Write to main memory for all internal cells
    writeBlock<w, h, gc_x, gc_y>( h1_ptr_,  h1_pitch_, Q[0], nx_, ny_, 0, 1);
    writeBlock<w, h, gc_x, gc_y>(hu1_ptr_, hu1_pitch_, Q[1], nx_, ny_, 0, 1);
    writeBlock<w, h, gc_x, gc_y>(hv1_ptr_, hv1_pitch_, Q[2], nx_, ny_, 0, 1);
    
    //Compute the CFL for this block
    if (cfl_ != NULL) {
        writeCfl<w, h, gc_x, gc_y, vars>(Q, F[0], nx_, ny_, dx_, dy_, g_, cfl_);
    }
}










} // extern "C"
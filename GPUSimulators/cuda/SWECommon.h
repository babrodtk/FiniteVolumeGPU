/*
These CUDA functions implement different types of numerical flux 
functions for the shallow water equations

Copyright (C) 2016, 2017, 2018 SINTEF Digital

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

#pragma once
#include "limiters.h"






__device__ float3 F_func(const float3 Q, const float g) {
    float3 F;

    F.x = Q.y;                              //hu
    F.y = Q.y*Q.y / Q.x + 0.5f*g*Q.x*Q.x;   //hu*hu/h + 0.5f*g*h*h;
    F.z = Q.y*Q.z / Q.x;                    //hu*hv/h;

    return F;
}













/**
  * Superbee flux limiter for WAF.
  * Related to superbee limiter so that WAF_superbee(r, c) = 1 - (1-|c|)*superbee(r)
  * @param r_ the ratio of upwind change (see Toro 2001, p. 203/204)
  * @param c_ the courant number for wave k, dt*S_k/dx
  */
__device__ float WAF_superbee(float r_, float c_) {
    // r <= 0.0
    if (r_ <= 0.0f) { 
        return 1.0f;
    }
    // 0.0 <= r <= 1/2
    else if (r_ <= 0.5f) { 
        return 1.0f - 2.0f*(1.0f - fabsf(c_))*r_;
    }
    // 1/2 <= r <= 1
    else if (r_ <= 1.0f) {
        return fabs(c_);
    }
    // 1 <= r <= 2
    else  if (r_ <= 2.0f) {
        return 1.0f - (1.0f - fabsf(c_))*r_;
    }
    // r >= 2
    else {
        return 2.0f*fabsf(c_) - 1.0f;
    }
}




__device__ float WAF_albada(float r_, float c_) {
    if (r_ <= 0.0f) {
        return 1.0f;
    }
    else {
        return 1.0f - (1.0f - fabsf(c_)) * r_ * (1.0f + r_) / (1.0f + r_*r_);
    }
}

__device__ float WAF_minbee(float r_, float c_) {
    r_ = fmaxf(-1.0f, fminf(2.0f, r_));
    if (r_ <= 0.0f) {
        return 1.0f;
    }
    if (r_ >= 0.0f && r_ <= 1.0f) {
        return 1.0f - (1.0f - fabsf(c_)) * r_;
    }
    else {
        return fabsf(c_);
    }
}

__device__ float WAF_minmod(float r_, float c_) {
    return 1.0f - (1.0f - fabsf(c_)) * fmaxf(0.0f, fminf(1.0f, r_));
}

__device__ float limiterToWAFLimiter(float r_, float c_) {
    return 1.0f - (1.0f - fabsf(c_))*r_;
}

// Compute h in the "star region", h^dagger
__device__ __inline__ float computeHStar(float h_l, float h_r, float u_l, float u_r, float c_l, float c_r, float g_) {
    
    //This estimate for the h* gives rise to spurious oscillations. 
    //return 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float h_tmp = 0.5f * (c_l + c_r) + 0.25f * (u_l - u_r);
    return h_tmp*h_tmp / g_;
}

/**
  * Weighted average flux (Toro 2001, p 200) for interface {i+1/2}
  * @param r_ The flux limiter parameter (see Toro 2001, p. 203)
  * @param Q_l2 Q_{i-1}
  * @param Q_l1 Q_{i}
  * @param Q_r1 Q_{i+1}
  * @param Q_r2 Q_{i+2}
  */
__device__ float3 WAF_1D_flux(const float3 Q_l2, const float3 Q_l1, const float3 Q_r1, const float3 Q_r2, const float g_, const float dx_, const float dt_) {     
    const float h_l = Q_l1.x;
    const float h_r = Q_r1.x;
    
    const float h_l2 = Q_l2.x;
    const float h_r2 = Q_r2.x;
    
    // Calculate velocities
    const float u_l = Q_l1.y / h_l;
    const float u_r = Q_r1.y / h_r;
    
    const float u_l2 = Q_l2.y / h_l2;
    const float u_r2 = Q_r2.y / h_r2;
    
    const float v_l = Q_l1.z / h_l;
    const float v_r = Q_r1.z / h_r;
    
    const float v_l2 = Q_l2.z / h_l2;
    const float v_r2 = Q_r2.z / h_r2;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    const float c_l2 = sqrt(g_*h_l2);
    const float c_r2 = sqrt(g_*h_r2);
    
    // Compute h in the "star region", h^dagger
    const float h_dag_l = computeHStar(h_l2,  h_l, u_l2,  u_l, c_l2,  c_l, g_);
    const float h_dag   = computeHStar( h_l,  h_r,  u_l,  u_r,  c_l,  c_r, g_);
    const float h_dag_r = computeHStar( h_r, h_r2,  u_r, u_r2,  c_r, c_r2, g_);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag ) ) / h_l;
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag ) ) / h_r;
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l; 
    const float S_r = u_r + c_r*q_r;
    const float S_star = ( S_l*h_r*(u_r - S_r) - S_r*h_l*(u_l - S_l) ) / ( h_r*(u_r - S_r) - h_l*(u_l - S_l) );
    
    const float3 Q_star_l = h_l * (S_l - u_l) / (S_l - S_star) * make_float3(1.0, S_star, v_l);
    const float3 Q_star_r = h_r * (S_r - u_r) / (S_r - S_star) * make_float3(1.0, S_star, v_r);
    
    // Estimate the fluxes in the four regions
    const float3 F_1 = F_func(Q_l1, g_);
    const float3 F_4 = F_func(Q_r1, g_);
    
    const float3 F_2 = F_1 + S_l*(Q_star_l - Q_l1);
    const float3 F_3 = F_4 + S_r*(Q_star_r - Q_r1);
    //const float3 F_2 = F_func(Q_star_l, g_);
    //const float3 F_3 = F_func(Q_star_r, g_);
    
    // Compute the courant numbers for the waves
    const float c_1 = S_l * dt_ / dx_;
    const float c_2 = S_star * dt_ / dx_;
    const float c_3 = S_r * dt_ / dx_;
    
    // Compute the "upwind change" vectors for the i-3/2 and i+3/2 interfaces
    const float eps = 1.0e-6f;
    const float r_1 = desingularize( (c_1 > 0.0f) ? (h_dag_l - h_l2) : (h_dag_r - h_r), eps) / desingularize((h_dag - h_l), eps);
    const float r_2 = desingularize( (c_2 > 0.0f) ? (v_l - v_l2) : (v_r2 - v_r), eps ) / desingularize((v_r - v_l), eps);
    const float r_3 = desingularize( (c_3 > 0.0f) ? (h_l - h_dag_l) : (h_r2 - h_dag_r), eps ) / desingularize((h_r - h_dag), eps);
        
    // Compute the limiter
    // We use h for the nonlinear waves, and v for the middle shear wave 
    const float A_1 = copysign(1.0f, c_1) * limiterToWAFLimiter(generalized_minmod(r_1, 1.9f), c_1);
    const float A_2 = copysign(1.0f, c_2) * limiterToWAFLimiter(generalized_minmod(r_2, 1.9f), c_2); 
    const float A_3 = copysign(1.0f, c_3) * limiterToWAFLimiter(generalized_minmod(r_3, 1.9f), c_3);
    
    //Average the fluxes
    const float3 flux = 0.5f*( F_1 + F_4 )
                      - 0.5f*( A_1 * (F_2 - F_1)
                             + A_2 * (F_3 - F_2)
                             + A_3 * (F_4 - F_3) );

    return flux;
}














/**
  * Central upwind flux function
  */
__device__ float3 CentralUpwindFlux(const float3 Qm, float3 Qp, const float g) {
    const float3 Fp = F_func(Qp, g);
    const float up = Qp.y / Qp.x;   // hu / h
    const float cp = sqrt(g*Qp.x); // sqrt(g*h)

    const float3 Fm = F_func(Qm, g);
    const float um = Qm.y / Qm.x;   // hu / h
    const float cm = sqrt(g*Qm.x); // sqrt(g*h)
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed
    
    return ((ap*Fm - am*Fp) + ap*am*(Qp-Qm))/(ap-am);
}









/**
  * Godunovs centered scheme (Toro 2001, p 165)
  */
__device__ float3 GodC_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    const float3 Q_godc = 0.5f*(Q_l + Q_r) + (dt_/dx_)*(F_l - F_r);
    
    return F_func(Q_godc, g_);
}
    


    
    




/**
  * Harten-Lax-van Leer with contact discontinuity (Toro 2001, p 180)
  */
__device__ float3 HLL_flux(const float3 Q_l, const float3 Q_r, const float g_) {    
    const float h_l = Q_l.x;
    const float h_r = Q_r.x;
    
    // Calculate velocities
    const float u_l = Q_l.y / h_l;
    const float u_r = Q_r.y / h_r;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    // Compute h in the "star region", h^dagger
    const float h_dag = 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag / (h_l*h_l) ) );
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag / (h_r*h_r) ) );
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l;
    const float S_r = u_r + c_r*q_r;
    
    //Upwind selection
    if (S_l >= 0.0f) {
        return F_func(Q_l, g_);
    }
    else if (S_r <= 0.0f) {
        return F_func(Q_r, g_);
    }
    //Or estimate flux in the star region
    else {
        const float3 F_l = F_func(Q_l, g_);
        const float3 F_r = F_func(Q_r, g_);
        const float3 flux = (S_r*F_l - S_l*F_r + S_r*S_l*(Q_r - Q_l)) / (S_r-S_l);
        return flux;
    }
}

















/**
  * Harten-Lax-van Leer with contact discontinuity (Toro 2001, p 181)
  */
__device__ float3 HLLC_flux(const float3 Q_l, const float3 Q_r, const float g_) {    
    const float h_l = Q_l.x;
    const float h_r = Q_r.x;
    
    // Calculate velocities
    const float u_l = Q_l.y / h_l;
    const float u_r = Q_r.y / h_r;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    // Compute h in the "star region", h^dagger
    const float h_dag = 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag / (h_l*h_l) ) );
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag / (h_r*h_r) ) );
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l;
    const float S_r = u_r + c_r*q_r;
    const float S_star = ( S_l*h_r*(u_r - S_r) - S_r*h_l*(u_l - S_l) ) / ( h_r*(u_r - S_r) - h_l*(u_l - S_l) );
    
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    //Upwind selection
    if (S_l >= 0.0f) {
        return F_l;
    }
    else if (S_r <= 0.0f) {
        return F_r;
    }
    //Or estimate flux in the "left star" region
    else if (S_l <= 0.0f && 0.0f <=S_star) {
        const float v_l = Q_l.z / h_l;
        const float3 Q_star_l = h_l * (S_l - u_l) / (S_l - S_star) * make_float3(1, S_star, v_l);
        const float3 flux = F_l + S_l*(Q_star_l - Q_l);
        return flux;
    }
    //Or estimate flux in the "righ star" region
    else if (S_star <= 0.0f && 0.0f <=S_r) {
        const float v_r = Q_r.z / h_r;
        const float3 Q_star_r = h_r * (S_r - u_r) / (S_r - S_star) * make_float3(1, S_star, v_r);
        const float3 flux = F_r + S_r*(Q_star_r - Q_r);
        return flux;
    }
    else {
        return make_float3(-99999.9f, -99999.9f, -99999.9f); //Something wrong here
    }
}













/**
  * Lax-Friedrichs flux (Toro 2001, p 163)
  */
__device__ float3 LxF_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    return 0.5f*(F_l + F_r) + (dx_/(2.0f*dt_))*(Q_l - Q_r);
}








/**
  * Lax-Friedrichs extended to 2D
  */
__device__ float3 LxF_2D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    //Note numerical diffusion for 2D here (0.25)
    return 0.5f*(F_l + F_r) + (dx_/(4.0f*dt_))*(Q_l - Q_r);
}












/**
  * Richtmeyer / Two-step Lax-Wendroff flux (Toro 2001, p 164)
  */
__device__ float3 LxW2_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    const float3 Q_lw2 = 0.5f*(Q_l + Q_r) + (dt_/(2.0f*dx_))*(F_l - F_r);
    
    return F_func(Q_lw2, g_);
}










 
    
/**
  * First Ordered Centered (Toro 2001, p.163)
  */
__device__ float3 FORCE_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_lf = LxF_1D_flux(Q_l, Q_r, g_, dx_, dt_);
    const float3 F_lw2 = LxW2_1D_flux(Q_l, Q_r, g_, dx_, dt_);
    return 0.5f*(F_lf + F_lw2);
}










template<int w, int h, int gc_x, int gc_y, int vars>
__device__ void writeCfl(float Q[vars][h+2*gc_y][w+2*gc_x],
        float shmem[h+2*gc_y][w+2*gc_x],
        const int nx_, const int ny_,
        const float dx_, const float dy_, const float g_,
        float* output_) {
    //Index of thread within block
    const int tx = threadIdx.x + gc_x;
    const int ty = threadIdx.y + gc_y;
    
    //Index of cell within domain
    const int ti = blockDim.x*blockIdx.x + tx;
    const int tj = blockDim.y*blockIdx.y + ty;
    
    //Only internal cells
    if (ti < nx_+gc_x && tj < ny_+gc_y) {
        const float h = Q[0][ty][tx];
        const float u   = Q[1][ty][tx] / h;
        const float v   = Q[2][ty][tx] / h;
        
        const float max_u = dx_ / (fabsf(u) + sqrtf(g_*h));
        const float max_v = dy_ / (fabsf(v) + sqrtf(g_*h));
        
        shmem[ty][tx] = fminf(max_u, max_v);
    }
    __syncthreads();
    
    //One row of threads loop over all rows
    if (ti < nx_+gc_x && tj < ny_+gc_y) {
        if (ty == gc_y) {
            float min_val = shmem[ty][tx];
            const int max_y = min(h, ny_+gc_y - tj);
            for (int j=gc_y; j<max_y+gc_y; j++) {
                min_val = fminf(min_val, shmem[j][tx]);
            }
            shmem[ty][tx] = min_val;
        }
    }
    __syncthreads();
    
    //One thread loops over first row to find global max
    if (tx == gc_x && ty == gc_y) {
        float min_val = shmem[ty][tx];
        const int max_x = min(w, nx_+gc_x - ti);
        for (int i=gc_x; i<max_x+gc_x; ++i) {
            min_val = fminf(min_val, shmem[ty][i]);
        }
        
        const int idx = gridDim.x*blockIdx.y + blockIdx.x;
        output_[idx] = min_val;
    }
}


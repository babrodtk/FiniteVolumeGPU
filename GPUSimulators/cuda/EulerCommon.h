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






inline __device__ float pressure(float4 Q, float gamma) {
    const float rho   = Q.x;
    const float rho_u = Q.y;
    const float rho_v = Q.z;
    const float E     = Q.w;

    return (gamma-1.0f)*(E-0.5f*(rho_u*rho_u + rho_v*rho_v)/rho);
}


__device__ float4 F_func(const float4 Q, float P) {
    const float rho   = Q.x;
    const float rho_u = Q.y;
    const float rho_v = Q.z;
    const float E     = Q.w;

    const float u = rho_u/rho;

    float4 F;

    F.x = rho_u;
    F.y = rho_u*u + P;
    F.z = rho_v*u;
    F.w = u*(E+P);

    return F;
}




/**
  * Harten-Lax-van Leer with contact discontinuity (Toro 2001, p 180)
  */
__device__ float4 HLL_flux(const float4 Q_l, const float4 Q_r, const float gamma) {
    const float h_l = Q_l.x;
    const float h_r = Q_r.x;
    
    // Calculate velocities
    const float u_l = Q_l.y / h_l;
    const float u_r = Q_r.y / h_r;
    
    // Calculate pressures
    const float P_l = pressure(Q_l, gamma);
    const float P_r = pressure(Q_r, gamma);
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(gamma*P_l/Q_l.x);
    const float c_r = sqrt(gamma*P_r/Q_r.x);
    
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
        return F_func(Q_l, P_l);
    }
    else if (S_r <= 0.0f) {
        return F_func(Q_r, P_r);
    }
    //Or estimate flux in the star region
    else {
        const float4 F_l = F_func(Q_l, P_l);
        const float4 F_r = F_func(Q_r, P_r);
        const float4 flux = (S_r*F_l - S_l*F_r + S_r*S_l*(Q_r - Q_l)) / (S_r-S_l);
        return flux;
    }
}







/**
  * Central upwind flux function
  */
__device__ float4 CentralUpwindFlux(const float4 Qm, const float4 Qp, const float gamma) {
    
    const float Pp = pressure(Qp, gamma);
    const float4 Fp = F_func(Qp, Pp);
    const float up = Qp.y / Qp.x;   // rho*u / rho
    const float cp = sqrt(gamma*Pp/Qp.x); // sqrt(gamma*P/rho)

    const float Pm = pressure(Qm, gamma);
    const float4 Fm = F_func(Qm, Pm);
    const float um = Qm.y / Qm.x;   // rho*u / rho
    const float cm = sqrt(gamma*Pm/Qm.x); // sqrt(gamma*P/rho)
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed
    
    return  ((ap*Fm - am*Fp) + ap*am*(Qp-Qm))/(ap-am);
}
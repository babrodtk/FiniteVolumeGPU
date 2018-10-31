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
  * Central upwind flux function
  */
__device__ float4 CentralUpwindFlux(const float4 Qm, float4 Qp, const float gamma) {
    
    const float Pp = pressure(Qp, gamma);
    const float4 Fp = F_func(Qp, Pp);
    const float up = Qp.y / Qp.x;   // rho*u / rho
    const float cp = sqrt(gamma*Pp*Qp.x); // sqrt(gamma*P/rho)

    const float Pm = pressure(Qm, gamma);
    const float4 Fm = F_func(Qm, Pm);
    const float um = Qm.y / Qm.x;   // rho*u / rho
    const float cm = sqrt(gamma*Pm/Qm.x); // sqrt(g*h)
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed
    
    return  ((ap*Fm - am*Fp) + ap*am*(Qp-Qm))/(ap-am);
}
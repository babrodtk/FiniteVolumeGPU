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

#pragma once






__device__ float pressure(float4 Q, float gamma) {
    const float rho   = Q.x;
    const float rho_u = Q.y;
    const float rho_v = Q.z;
    const float E     = Q.w;

    return (gamma-1.0f)*(E-0.5f*(rho_u*rho_u + rho_v*rho_v)/rho);
}


__device__ float4 F_func(const float4 Q, float gamma) {
    const float rho   = Q.x;
    const float rho_u = Q.y;
    const float rho_v = Q.z;
    const float E     = Q.w;

    const float u = rho_u/rho;
    const float P = pressure(Q, gamma);

    float4 F;

    F.x = rho_u;
    F.y = rho_u*u + P;
    F.z = rho_v*u;
    F.w = u*(E+P);

    return F;
}
/*
This file implements the Harten-Lax-van Leer flux with contact discontinuity

Copyright (C) 2016, 2017, 2018 SINTEF ICT

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


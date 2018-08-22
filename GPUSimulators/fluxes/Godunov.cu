/*
This file implements the Godunov flux

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
  * Godunovs centered scheme (Toro 2001, p 165)
  */
__device__ float3 GodC_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    const float3 Q_godc = 0.5f*(Q_l + Q_r) + (dt_/dx_)*(F_l - F_r);
    
    return F_func(Q_godc, g_);
}
    



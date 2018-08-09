/*
This file implements the First ORder CEntered flux

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

#include "./LaxFriedrichs.cu"
#include "./LaxWendroff.cu"


    
    
/**
  * First Ordered Centered (Toro 2001, p.163)
  */
__device__ float3 FORCE_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_lf = LxF_1D_flux(Q_l, Q_r, g_, dx_, dt_);
    const float3 F_lw2 = LxW2_1D_flux(Q_l, Q_r, g_, dx_, dt_);
    return 0.5f*(F_lf + F_lw2);
}

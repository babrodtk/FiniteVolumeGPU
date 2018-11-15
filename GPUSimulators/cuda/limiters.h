/*
This file implements different flux and slope limiters

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

#pragma once






/**
  * Reconstructs a slope using the generalized minmod limiter based on three 
  * consecutive values
  */
__device__ __inline__ float minmodSlope(float left, float center, float right, float theta) {
    const float backward = (center - left) * theta;
    const float central = (right - left) * 0.5f;
    const float forward = (right - center) * theta;
    
	return 0.25f
		*copysign(1.0f, backward)
		*(copysign(1.0f, backward) + copysign(1.0f, central))
		*(copysign(1.0f, central) + copysign(1.0f, forward))
		*min( min(fabs(backward), fabs(central)), fabs(forward) );
}




/**
  * Reconstructs a minmod slope for a whole block along the abscissa
  */
template<int w, int h, int gc_x, int gc_y, int vars>
__device__ void minmodSlopeX(float Q[vars][h+2*gc_y][w+2*gc_x],
                  float Qx[vars][h+2*gc_y][w+2*gc_x],
                  const float theta_) {
    //Reconstruct slopes along x axis
    for (int p=0; p<vars; ++p) {
        for (int j=threadIdx.y; j<h+2*gc_y; j+=h) {
            for (int i=threadIdx.x+1; i<w+2*gc_x-1; i+=w) {
                Qx[p][j][i] = minmodSlope(Q[p][j][i-1], Q[p][j][i], Q[p][j][i+1], theta_);
            }
        }
    }
}


/**
  * Reconstructs a minmod slope for a whole block along the ordinate
  */
template<int w, int h, int gc_x, int gc_y, int vars>
__device__ void minmodSlopeY(float Q[vars][h+2*gc_y][w+2*gc_x],
                  float Qy[vars][h+2*gc_y][w+2*gc_x],
                  const float theta_) {
    //Reconstruct slopes along y axis
    for (int p=0; p<vars; ++p) {
        for (int j=threadIdx.y+1; j<h+2*gc_y-1; j+=h) {
            for (int i=threadIdx.x; i<w+2*gc_x; i+=w) {
                Qy[p][j][i] = minmodSlope(Q[p][j-1][i], Q[p][j][i], Q[p][j+1][i], theta_);
            }
        }
    }
}




__device__ float monotonized_central(float r_) {
    return fmaxf(0.0f, fminf(2.0f, fminf(2.0f*r_, 0.5f*(1.0f+r_))));
}

__device__ float osher(float r_, float beta_) {
    return fmaxf(0.0f, fminf(beta_, r_));
}

__device__ float sweby(float r_, float beta_) {
    return fmaxf(0.0f, fmaxf(fminf(r_, beta_), fminf(beta_*r_, 1.0f)));
}

__device__ float minmod(float r_) {
    return fmaxf(0.0f, fminf(1.0f, r_));
}

__device__ float generalized_minmod(float r_, float theta_) {
    return fmaxf(0.0f, fminf(theta_*r_, fminf( (1.0f + r_) / 2.0f, theta_)));
}

__device__ float superbee(float r_) {
    return fmaxf(0.0f, fmaxf(fminf(2.0f*r_, 1.0f), fminf(r_, 2.0f)));
}

__device__ float vanAlbada1(float r_) {
    return (r_*r_ + r_) / (r_*r_ + 1.0f);
}

__device__ float vanAlbada2(float r_) {
    return 2.0f*r_ / (r_*r_* + 1.0f);
}

__device__ float vanLeer(float r_) {
    return (r_ + fabsf(r_)) / (1.0f + fabsf(r_));
}
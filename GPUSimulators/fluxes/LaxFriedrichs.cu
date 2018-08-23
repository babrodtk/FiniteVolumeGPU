

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

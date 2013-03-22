# -*- coding: utf-8 -*-

"""Functions computing derived quantities from ROMS output"""

def kinetic_energy(U,V):
    """Kinetic energy from velocities in a C-grid

    U : U-component, velocity [m/s], shape[-2:] = (jmax, imax+1)
    V : V-component, velocity [m/s], shape[-2:] = (jmax+1, imax+1)
    
    returns average kinetic energy KE in the grid cell,
    without the density factor

    KE : shape[-2:] = (jmax, imax), [J/kg]
    
    """

    # Uses estimate (U_w**2 + U_e**2 + U_w*U_e)/3 for U**2
    # This is correct to second order
    
    KE = (  U[...,:,:-1]**2 + U[...,:,1:]**2 + U[...,:,:-1]*U[...,:,1:]
          + V[...,:-1,:]**2 + V[...,1:,:]**2 + V[...,:-1,:]*V[...,1:,:]) / 6.0
    
    return KE


# -*- coding: utf-8 -*-

"""Functions computing derived quantities from ROMS output"""

def kinetic_energy(U, V):
    """Kinetic energy from velocities in a C-grid

    U : U-component, velocity [m/s], shape[-2:] = (jmax, imax+1)
    V : V-component, velocity [m/s], shape[-2:] = (jmax+1, imax+1)
    
    returns average kinetic energy KE in the grid cell,
    without the density factor

    KE : shape[-2:] = (jmax, imax), [J/kg] = [m**2/s**2]
    
    """

    # Uses estimate (U_w**2 + U_e**2 + U_w*U_e)/3 for U**2
    # This is correct to second order
    
    KE = (  U[...,:,:-1]**2 + U[...,:,1:]**2 + U[...,:,:-1]*U[...,:,1:]
          + V[...,:-1,:]**2 + V[...,1:,:]**2 + V[...,:-1,:]*V[...,1:,:]) / 6.0
    
    return KE

# ---------

# Note,
# To integrate the flux out of a cell
# we can have U(i-0.5) = U(i+0.5) but
# if the pm (or dy) is averaged with neighbour
# we can have 1/pm(i+0.5) unlike 1/pm(i-0.5)
# which gives a non-zero flux in/out of the cell.
# Generell krumlinje-divergens, hvordan?

def divergence(U, V, pm, pn):
    """Divergence from C-grid velocity

    U : U-component, velocity [m/s], shape[-2:] = (jmax, imax+1)
    V : V-component, velocity [m/s], shape[-2:] = (jmax+1, imax+1)
    pm : invers metric term X-direction [1/m], shape = (jmax, imax)
    p : invers metric term X-direction [1/m], shape = (jmax, imax)

    Result:
    div : divergence [1/s]],  shape[-2:] = (jmax, imax), [1/s]
    
    """

    div = ( (U[...,:,1:] - U[...,:,:-1])*pm +
            (V[...,1:,:] - V[...,:-1,:])*pn )

    return div

# -------------

def div2(U, V, pm, pn):
    """Divergence from C-grid velocity

    U : U-component, velocity [m/s], shape[-2:] = (jmax, imax+1)
    V : V-component, velocity [m/s], shape[-2:] = (jmax+1, imax+1)
    pm : invers metric term X-direction [1/m], shape = (jmax+2, imax+2)
    pn : invers metric term Y-direction [1/m], shape = (jmax+2, imax+2)

    Result:
    div : divergence [1/s],  shape[-2:] = (jmax, imax), [1/s]
    
    """

    # pm*pn*(d/dx(U/pn) - d/dt(V/pm))
    om_u = 2.0 / (pn[1:-1, :-1] + pn[1:-1, 1:])
    on_v = 2.0 / (pm[:-1, 1:-1] + pm[1:, 1:-1])
    div = pm * pn * (
             (U[...,:,1:] - U[...,:,:-1]) * om_u
           + (V[...,1:,:] - V[...,:-1,:]) * on_v)

    return div


# -*- coding: utf-8 -*-

"""Functions computing derived quantities from ROMS output"""


def kinetic_energy(U, V):
    """Kinetic energy from velocities in a C-grid

    U : U-component, velocity [m/s], shape[-2:] = (jmax, imax+1)
    V : V-component, velocity [m/s], shape[-2:] = (jmax+1, imax)

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


# -------------

# Could have pm.shape = (jmax, imax+2)
#            pn.shape = (jmax+2, imax)


# Ta pm, pn optional:
# pn mangler: pn = pm
# pm mangler, begge = 1.0
# Også godta skalare pm = 1.0/dx, pn = 1.0/dy = constant.


def divergence(U, V, pm, pn):
    """Divergence from C-grid velocity

    U : U-component, velocity [m/s], shape[-2:] = (jmax, imax+1)
    V : V-component, velocity [m/s], shape[-2:] = (jmax+1, imax)
    pm : invers metric term X-direction [1/m], shape = (jmax+2, imax+2)
    pn : invers metric term Y-direction [1/m], shape = (jmax+2, imax+2)

    Result:
    div : divergence [1/s],  shape[-2:] = (jmax, imax), [1/s]

    The divergence for an orthogonal coordinate system is given by::

      div = pm * pn * (d/dx(U/pn) + d/dy(V/pm))

    """

    A = 2.0 * U / (pn[1:-1, :-1] + pn[1:-1, 1:])    # U/pn
    B = 2.0 * V / (pm[:-1, 1:-1] + pm[1:, 1:-1])    # V/pm
    pmn = pm[1:-1, 1:-1] * pn[1:-1, 1:-1]

    div = pmn * (  A[..., :, 1:] - A[..., :, :-1]
                 + B[..., 1:, :] - B[..., :-1, :])

    return div

# ---------


def curl(U, V, pm, pn):
    """2D curl from C-grid velocity

    rho-points have shape (jmax, imax)
    U : U-component, velocity [m/s], shape[-2:] = (jmax, imax-1)
    V : V-component, velocity [m/s], shape[-2:] = (jmax-1, imax)
    pm : invers metric term X-direction [1/m], shape = (jmax, imax)
    pn : invers metric term Y-direction [1/m], shape = (jmax, imax)

    Result:
    curl [1/s],  shape[-2:] = (jmax-1, imax-1) at internal psi-points

    The divergence for an orthogonal coordinate system is given by::

      curl = pm * pn * (d/dy(U/pm) - d/dx(V/pn))

    """

    # Problem: midle pm eller midle dx = 1/pm ????
    # Hva er best numerisk

    # Factors at psi-points
    pm_psi = 0.25*(pm[:-1,:-1] + pm[1:,:-1] + pm[:-1,1:] + pm[1:,1:])
    pn_psi = 0.25*(pn[:-1,:-1] + pn[1:,:-1] + pn[:-1,1:] + pn[1:,1:])

    # Denne gir korrekt på solid body i polar grid
    # (men alltid bedre ???)
    # pm_psi = 4.0/(1/pm[:-1,:-1] + 1/pm[1:,:-1] + 1/pm[:-1,1:] + 1/pm[1:,1:])
    # pn_psi = 4.0/(1/pn[:-1,:-1] + 1/pn[1:,:-1] + 1/pn[:-1,1:] + 1/pn[1:,1:])

    pmn = pm_psi * pn_psi

    Udx = 2 * U / (pm[:, :-1] + pm[:, 1:])     # U/pm
    Vdy = 2 * V / (pn[:-1, :] + pn[1:, :])     # V/pn

    curl = pmn * (   Udx[..., 1:, :] - Udx[..., :-1, :]
                   - Vdy[..., :, 1:] + Vdy[..., :, :-1])

    return curl

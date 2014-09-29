# -*- coding: utf-8 -*-

# Section class for flux calculations

import numpy as np

from depth import sdepth
from sample import sample2D


class FluxSection(object):
    """Class for flux calculations across sections

    A FluxSection object is defined by a sequence of psi-points

    psi-points are indexed using ROMS convention (one-based)
    psi(i,j) = psi[j-1, i-1] lives at x=i-0.5, y=j-0.5


    NOTE: not yet defined if psi-point (i,j) lives at
    x = i-0.5 or x = i+0.5 and similar for y
    Follow the convention of pyroms (check)

    The grid information is defined by a grid object having attributes

       h, pm, pn, hc, Cs_r, Cs_w, Vtransform

    with the ROMS variables of the same netCDF names

    psi(i,j) = psi[j-1,i-1] lives at i-0.5, i+0.5

    """

    def __init__(self, grid, I, J):

        self.grid = grid
        self.I = np.asarray(I)
        self.J = np.asarray(J)
        # Grid coordinates of (mid points of) edges
        self.X = 0.5*(self.I[:-1]+self.I[1:]) - 0.5
        self.Y = 0.5*(self.J[:-1]+self.J[1:]) - 0.5

        # Logical indexing for U, V edges
        E = np.arange(len(self.I)-1, dtype=int)  # Edge indices
        self.Eu = (self.I[:-1] == self.I[1:])  # True for U edges
        self.Ev = (self.J[:-1] == self.J[1:])  # True for V edges

        # Section size
        self.L = len(self.I)-1         # Number of nodes
        self.N = len(self.grid.Cs_r)

        # Topography
        # Could simplify, since sampling here is
        # simply averaging of two values
        self.h = sample2D(self.grid.h, self.X, self.Y,
                          mask=self.grid.mask_rho)

        # Metric
        pm = sample2D(self.grid.pm, self.X, self.Y)
        pn = sample2D(self.grid.pn, self.X, self.Y)

        # Distance along section
        dX = np.where(self.Ev, 1.0/pm, 0)
        dY = np.where(self.Eu, 1.0/pn, 0)
        # dS = sqrt(dX**2 + dY**2) simplifies in this case
        self.dS = np.maximum(dX, dY)
        self.dX, self.dY = dX, dY

        # Vertical structure
        self.z_w = sdepth(self.h, self.grid.hc, self.grid.Cs_w,
                          stagger='w', Vtransform=self.grid.Vtransform)
        self.dZ = self.z_w[1:, :]-self.z_w[:-1, :]
        self.dSdZ = self.dS * self.dZ

        # Direction
        # Convention, positive flux to the right of the sequence
        # U-edge: pointing up, positive flux right,   dir = +1
        #         pointing down, positive lux left,   dir = -1
        # V-edge: pointing right, positive flux down, dir = -1
        #         pointing left, positive flux up,    dir = +1
        dir = np.zeros((self.L,), dtype=int)
        dir[self.Eu] =   self.J[1:][self.Eu] - self.J[:-1][self.Eu]
        dir[self.Ev] = - self.I[1:][self.Ev] + self.I[:-1][self.Ev]
        self.dir = dir

    def __len__(self):
        return self.L

    def transport(self, U, V):
        #
        # if (jmax, imax) = shape(grid.h)
        # must have: shape(U) = (jmax, imax-1)
        #            shape(V) = (jmax-1, imax)
        #        U = np.zeros((kmax, jmax, imax-1))

        # U-edge, from psi(i,j) -> psi(i,j+1), dir=+1
        #       ROMS: u(i,j), python: u[j, i-1],
        # U-edge, from psi(i,j) -> psi(i, j-1), dir=-1
        #       ROMS: u(i, j-1), python: u[j-1, i-1],
        # In general:
        #   python: u[j-(1-dir)//2, i-1]

        # V-edge, psi(i,j) -> psi(i+1,j), dir=-1
        # dir have opposite role, use (1+dir)//2

        dirU = self.dir[self.Eu]
        dirV = self.dir[self.Ev]
        IU = self.I[self.Eu]
        IV = self.I[self.Ev]
        JU = self.J[self.Eu]
        JV = self.J[self.Ev]

        Usec = dirU * U[:, JU - (1-dirU)//2, IU - 1]
        Vsec = dirV * V[:, JV - 1, IV - (1+dirV)//2]

        UVsec = np.zeros((self.N, self.L))
        UVsec[:, self.Eu] = Usec
        UVsec[:, self.Ev] = Vsec

        Flux = np.sum(UVsec * self.dSdZ)
        M = UVsec > 0
        Flux_plus = np.sum(UVsec[M] * self.dSdZ[M])

        return Flux, Flux_plus
    
# -------------------------------------------


def staircase_from_line(i0, i1, j0, j1):

    swapXY = False
    if abs(i1-i0) < abs(j0-j1): # Mostly vertical
        i0, i1, j0, j1 = j0, j1, i0, i1
        swapXY = True

    # Find integer points X0 and Y0 on line
    if i0 < i1:
        X0 = list(range(i0, i1+1))
    elif i0 > i1:
        X0 = list(range(i0, i1-1, -1))
    else:  # i0 = i1 and j0 = j1
        raise ValueError("Section reduced to a point")
    slope = float(j1-j0) / (i1-i0)
    Y0 = [j0 + slope*(x-i0) for x in X0]

    # sign = -1 if Y0 is decreasing, otherwise sign = 1
    sign = 1
    if Y0[-1] < Y0[0]:     # Decreasing Y
        sign = -1

    # Make lists of positions along staircase
    X, Y = [i0], [j0]

    for i in range(len(X0)-1):
        x, y = X[-1], Y[-1]          # Last point on list
        x0, y0 = X0[i], Y0[i]        # Present point along line
        x1, y1 = X0[i+1], Y0[i+1]    # Next point along line
        if abs(y - y1) > 0.5:        # Append extra point
            if sign*(y - y0) < 0:        # Jump first
                X.append(x0)
                Y.append(y+sign)
                X.append(x1)
                Y.append(y+sign)
            else:                        # Jump last
                X.append(x1)
                Y.append(y)
                X.append(x1)
                Y.append(y+sign)
        else:                        # Ordinary append
            X.append(x1)
            Y.append(y)

    if swapXY:
        X, Y = Y, X

    return np.array(X), np.array(Y)

# -*- coding: utf-8 -*-

# Section class for flux calculations

from __future__ import division, absolute_import
import numpy as np

from roppy.depth import sdepth


class FluxSection(object):
    """Class for flux calculations across sections

    A FluxSection object is defined by a sequence of psi-points
    following a staircase curve

    psi-points are indexed using ROMS convention (one-based)
    psi(i,j) = psi[j-1, i-1] lives at x=i-0.5, y=j-0.5

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
        self.X = 0.5 * (self.I[:-1] + self.I[1:]) - 0.5
        self.Y = 0.5 * (self.J[:-1] + self.J[1:]) - 0.5

        # Section size
        self.L = len(self.I) - 1  # Number of nodes
        self.N = len(self.grid.Cs_r)

        # Logical indexing for U, V edges
        # E = np.arange(self.L, dtype=int)       # Edge indices
        self.Eu = self.I[:-1] == self.I[1:]  # True for U edges
        self.Ev = self.J[:-1] == self.J[1:]  # True for V edges

        # Direction
        # Convention, positive flux to the right of the sequence
        # U-edge: pointing up, positive flux right,   dir = +1
        #         pointing down, positive lux left,   dir = -1
        # V-edge: pointing right, positive flux down, dir = -1
        #         pointing left, positive flux up,    dir = +1
        dir = np.zeros((self.L,), dtype=int)
        dir[self.Eu] = self.J[1:][self.Eu] - self.J[:-1][self.Eu]
        dir[self.Ev] = -self.I[1:][self.Ev] + self.I[:-1][self.Ev]
        self.dir = dir

        # Topography
        # Could simplify, since sampling here is
        # simply averaging of two values
        self.h = self.sample2D(self.grid.h)

        # Metric
        pm = self.sample2D(self.grid.pm)
        pn = self.sample2D(self.grid.pn)

        # Distance along section
        dX = np.where(self.Ev, 1.0 / pm, 0)
        dY = np.where(self.Eu, 1.0 / pn, 0)
        # dS = sqrt(dX**2 + dY**2) simplifies in this case
        self.dS = np.maximum(dX, dY)
        self.dX, self.dY = dX, dY

        # Vertical structure
        self.z_w = sdepth(
            self.h,
            self.grid.hc,
            self.grid.Cs_w,
            self.grid.s_w,
            stagger="w",
            Vtransform=self.grid.Vtransform,
            Vstretching = self.grid.Vstretching,
        )
        self.dZ = self.z_w[1:, :] - self.z_w[:-1, :]
        self.dSdZ = self.dS * self.dZ

    def __len__(self):
        return self.L

    def flux_array(self, U, V):
        """Returns a 2D field of fluxes through the section"""

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

        # A subgrid has velocity components at the boundaries
        # giving an extra velocity offset
        ioff, joff = 0, 0
        if self.grid.i0 > 0:
            ioff = 1
        if self.grid.j0 > 0:
            joff = 1

        I = self.I[:-1]
        J = self.J[:-1]

        IU = I[self.Eu] - self.grid.i0 - 1 - ioff
        IV = I[self.Ev] - (1 + dirV) // 2 - self.grid.i0
        JU = J[self.Eu] - (1 - dirU) // 2 - self.grid.j0
        JV = J[self.Ev] - self.grid.j0 - 1 - joff

        UVsec = np.empty((self.N, self.L))
        UVsec[:, self.Eu] = dirU * U[:, JU, IU]
        UVsec[:, self.Ev] = dirV * V[:, JV, IV]

        return UVsec * self.dSdZ

    # -------------------------------

    def transport(self, U, V):
        """Integrated flux though the section"""

        Flux = self.flux_array(U, V)
        tot_flux = np.sum(Flux)
        pos_flux = np.sum(Flux[Flux > 0])

        return tot_flux, pos_flux

    # ---------------------------------

    def sample2D(self, F):
        """Sample a horizontal field (rho-points) to the section edges"""

        # Could simplify since average og two neighbouring rho-cells
        # return sample2D(F, self.X, self.Y)

        dirU = self.dir[self.Eu]
        dirV = self.dir[self.Ev]

        # Find indices
        I = self.I[:-1]
        J = self.J[:-1]
        IU = I[self.Eu] - self.grid.i0
        IV = I[self.Ev] - (1 + dirV) // 2 - self.grid.i0
        JU = J[self.Eu] - (1 - dirU) // 2 - self.grid.j0
        JV = J[self.Ev] - self.grid.j0

        # Average F to U- and V-points
        Fsec = np.empty((self.L,), F.dtype)
        Fsec[self.Eu] = 0.5 * (F[JU, IU] + F[JU, IU - 1])
        Fsec[self.Ev] = 0.5 * (F[JV, IV] + F[JV - 1, IV])

        return Fsec

    # ---------------------------------

    def sample3D(self, F):
        """Sample a 3D (rho-)field to the section"""

        # Could be simplified bu sample2D above

        Fsec = np.zeros((self.N, self.L))
        for k in range(self.grid.N):
            Fsec[k, :] = self.sample2D(F[k, :, :])
        return Fsec


# -------------------------------------------


def staircase_from_line(i0, i1, j0, j1):

    swapXY = False
    if abs(i1 - i0) < abs(j0 - j1):  # Mostly vertical
        i0, i1, j0, j1 = j0, j1, i0, i1
        swapXY = True

    # Find integer points X0 and Y0 on line
    if i0 < i1:
        X0 = list(range(i0, i1 + 1))
    elif i0 > i1:
        X0 = list(range(i0, i1 - 1, -1))
    else:  # i0 = i1 and j0 = j1
        raise ValueError("Section reduced to a point")
    slope = float(j1 - j0) / (i1 - i0)
    Y0 = [j0 + slope * (x - i0) for x in X0]

    # sign = -1 if Y0 is decreasing, otherwise sign = 1
    sign = 1
    if Y0[-1] < Y0[0]:  # Decreasing Y
        sign = -1

    # Make lists of positions along staircase
    X, Y = [i0], [j0]

    for i in range(len(X0) - 1):
        x, y = X[-1], Y[-1]  # Last point on list
        x0, y0 = X0[i], Y0[i]  # Present point along line
        x1, y1 = X0[i + 1], Y0[i + 1]  # Next point along line
        if abs(y - y0) + abs(y - y1) > abs(y + sign - y0) + abs(y + sign - y1):
            # jump
            X.append(x0)
            Y.append(y + sign)
            X.append(x1)
            Y.append(y + sign)
        else:  # Ordinary append
            X.append(x1)
            Y.append(y)
    # Possible jump to last point
    # Assumes Y0[-1]
    if Y[-1] != j1:
        X.append(i1)
        Y.append(j1)

    if swapXY:
        X, Y = Y, X

    return np.array(X), np.array(Y)

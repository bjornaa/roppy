# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
from roppy.sample import sample2D, sample2DU, sample2DV
from roppy.depth import sdepth


class Section(object):
    """Class for handling sections in a ROMS grid

    The grid is defined by a grid object having attributes

       h, pm, pn, hc, Cs_r, Cs_w, Vtransform

    with the ROMS variables of the same netCDF names

    Defined by sequences of grid coordinates of vertices

    """

    def __init__(self, grid, X, Y):

        self.grid = grid
        # Vertices, in subgrid coordinates
        self.X = X - grid.i0
        self.Y = Y - grid.j0

        # Nodes
        self.Xm = 0.5 * (self.X[:-1] + self.X[1:])
        self.Ym = 0.5 * (self.Y[:-1] + self.Y[1:])

        # Section size
        self.nseg = len(self.Xm)  # Number of segments = Number of nodes
        self.N = len(self.grid.Cs_r)

        # Compatible with both SGrid and grdClass
        try:
            self.h = sample2D(
                self.grid.h, self.Xm, self.Ym, mask=self.grid.mask_rho, undef_value=0.0
            )
        except AttributeError:
            self.h = sample2D(self.grid.depth, self.Xm, self.Ym)

        pm = sample2D(self.grid.pm, self.Xm, self.Ym)
        pn = sample2D(self.grid.pn, self.Xm, self.Ym)

        # Unit normal vector (nx, ny)
        # Sjekk om dette er korrekt hvis pm og pn er ulike
        dX = (X[1:] - X[:-1]) / pm
        dY = (Y[1:] - Y[:-1]) / pn

        # Length of segments
        #   Kan kanskje forbedres med sfærisk avstand
        self.dS = np.sqrt(dX * dX + dY * dY)
        # Cumulative distance (at vertices)s
        self.S = np.concatenate(([0], np.add.accumulate(self.dS)))

        nx, ny = dY, -dX
        norm = np.sqrt(nx * nx + ny * ny)
        self.nx, self.ny = nx / norm, ny / norm

        # Vertical structure
        self.z_r = sdepth(
            self.h,
            self.grid.hc,
            self.grid.Cs_r,
            stagger="rho",
            Vtransform=self.grid.Vtransform,
        )
        self.z_w = sdepth(
            self.h,
            self.grid.hc,
            self.grid.Cs_w,
            stagger="w",
            Vtransform=self.grid.Vtransform,
        )
        self.dZ = self.z_w[1:, :] - self.z_w[:-1, :]

        self.Area = self.dZ * self.dS

    def sample2D(self, F):
        return sample2D(F, self.Xm, self.Ym)

    def sample3D(self, F):
        """Sample a 3D field in rho-points with shape (N,Mp,Lp)"""

        # Interpolerer foreløpig langs s-flater
        # Sikkert OK for plotting, Godt nok for flux-beregning?

        Fsec = np.zeros((self.grid.N, self.nseg))
        for k in range(self.grid.N):
            Fsec[k, :] = sample2D(F[k, :, :], self.Xm, self.Ym, mask=self.grid.mask_rho)
        Fsec = np.ma.masked_where(self.extend_vertically(self.h) == 0, Fsec)
        return Fsec

    def normal_current(self, U, V):
        """Sample normal component of velocity field"""

        # Interpolerer foreløpig langs s-flater

        # Offset for interpolation from U and V grid
        deltaU = -0.5 + self.grid.i0 - self.grid.i0_u
        deltaV = -0.5 + self.grid.j0 - self.grid.j0_v
        Usec = np.zeros((self.N, self.nseg))
        Vsec = np.zeros((self.N, self.nseg))
        for k in range(self.N):
            Usec[k, :] = sample2D(U[k, :, :], self.Xm + deltaU, self.Ym)
            Vsec[k, :] = sample2D(V[k, :, :], self.Xm, self.Ym + deltaV)
        return self.nx * Usec + self.ny * Vsec

    def extend_vertically(self, F):
        """extends a 1D array to all s-levels"""
        return np.outer(np.ones(self.N), F)

    def Flux(self, U, V):
        """Returns 2D volume flux array"""
        return self.Area * self.normal_current(U, V) * 1.0e-6

    def flux(self, U, V, mask=None):
        """Compute net volume flux across the section

        U, V are 3D current fields,
        if mask: only computes flux where True

        returns flux in Sverdrup, positive is net flux trough
        in the right direction of the section

        """

        Flux = self.Area * self.normal_current(U, V)
        if mask is not None:
            Flux = Flux[mask]
        return np.sum(Flux) * 1.0e-6  # Convert to Sverdrup

    # Room for improvement, cache the normal_current?
    def flux_r(self, U, V, mask=None):
        """Compute volume flux across the section towards right

        U, V are 3D current fields,
        if mask: only computes flux where True

        returns flux in Sverdrup

        """

        Flux = self.Area * self.normal_current(U, V)
        if mask is not None:
            Flux = Flux[mask]
        return np.sum(Flux[Flux > 0]) * 1.0e-6  # Convert to Sverdrup

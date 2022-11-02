import numpy as np
from numpy.typing import NDArray

from roppy.sgrid import SGrid
from roppy.sample import sample2D
from roppy.depth import sdepth


class Section:
    """Class for handling sections in a ROMS grid

    The section is defined by a sequence of nodes, supposedly quite close
    The endpoints of the section are nodes

    The grid information is defined by a grid object having attributes

       h, pm, pn, hc, Cs_r, Cs_w, Vtransform

    with the ROMS variables of the same netCDF names

    Defined by sequences of grid coordinates of section nodes

    """

    def __init__(self, grid: SGrid, X: NDArray[np.float64], Y: NDArray[np.float64]):

        self.grid = grid
        # Vertices, in subgrid coordinates
        self.X = X
        self.Y = Y

        # Section size
        self.L = len(self.X)  # Number of nodes
        self.N = len(self.grid.Cs_r)

        # Topography
        self.h: NDArray[np.float64] = sample2D(
            self.grid.h, self.X, self.Y, mask=self.grid.mask_rho, undef_value=1.0  # type: ignore
        )

        # Metric
        pm: NDArray[np.float64] = sample2D(self.grid.pm, self.X, self.Y)  # type: ignore
        pn: NDArray[np.float64] = sample2D(self.grid.pn, self.X, self.Y)  # type: ignore
        dX = 2 * (X[1:] - X[:-1]) / (pm[:-1] + pm[1:])  # unit = meter
        dY = 2 * (Y[1:] - Y[:-1]) / (pn[:-1] + pn[1:])
        # Assume spacing is close enough to approximate distance
        self.dS = np.sqrt(dX * dX + dY * dY)
        # Cumulative distance
        self.S = np.concatenate(([0], np.add.accumulate(self.dS)))  # type: ignore
        # Weights for trapez integration (linear interpolation)
        self.W = 0.5 * np.concatenate(
            ([self.dS[0]], self.dS[:-1] + self.dS[1:], [self.dS[-1]])
        )

        # nx, ny  = dY, -dX
        # norm = np.sqrt(nx*nx + ny*ny)
        # self.nx, self.ny = nx/norm, ny/norm

        # Vertical structure
        self.z_r = sdepth(
            self.h,  # type: ignore
            self.grid.hc,
            self.grid.Cs_r,
            self.grid.s_rho,
            stagger="rho",
            Vtransform=self.grid.Vtransform,
        )
        self.z_w = sdepth(
            self.h,  # type: ignore
            self.grid.hc,
            self.grid.Cs_w,
            self.grid.s_w,
            stagger="w",
            Vtransform=self.grid.Vtransform,
        )
        self.dZ = self.z_w[1:, :] - self.z_w[:-1, :]
        self.Area = self.dZ * self.W

    def __len__(self) -> int:
        return self.L

    def sample2D(self, F: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample a horizontal field at rho poins with shape (Mp, Lp)"""
        return sample2D(F, self.X, self.Y, mask=self.grid.mask_rho)  # type: ignore

    def sample3D(self, F: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample a 3D field in rho-points with shape (N,Mp,Lp)"""

        # Not masked ??

        Fsec = np.zeros((self.N, self.L))
        for k in range(self.N):
            Fsec[k, :] = sample2D(
                F[k, :, :], self.X, self.Y, mask=self.grid.mask_rho  # type: ignore
            )
        return Fsec


def linear_section(i0: int, i1: int, j0: int, j1: int, grd: SGrid) -> Section:
    """Make a linear section between rho-points

    Makes a section similar to romstools' tools.transect

    Returns a section object
    """

    if abs(i1 - i0) >= abs(j0 - j1):  # Work horizontally
        if i0 < i1:
            X = np.arange(i0, i1 + 1, dtype=np.float64)
        elif i0 > i1:
            X = np.arange(i0, i1 - 1, -1, dtype=np.float64)
        else:  # i0 = i1 and j0 = j1
            raise ValueError("Section reduced to a point")
        slope = float(j1 - j0) / (i1 - i0)
        Y = j0 + slope * (X - i0)

    else:  # Work vertically
        if j0 < j1:
            Y = np.arange(j0, j1 + 1, dtype=np.float64)
        else:
            Y = np.arange(j0, j1 - 1, -1, dtype=np.float64)
        slope = float(i1 - i0) / (j1 - j0)
        X = i0 + slope * (Y - j0)

    return Section(grd, X, Y)

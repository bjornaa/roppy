# -*- coding: utf-8 -*-

"""Generator for moving averages from ROMS file(s)"""

import numpy as np


def roms_averator(ncid, var_name, L, grd):
    """Generator for moving averages from ROMS file(s)

    var_name : text string, name of NetCDF variable

    ncid : an open NetCDF Dataset or MFDataset

    grd : a roppy.SGrid instance

    L : integer, length of averaging period (only even presently)

    n_rec = len(fid.dimensions['ocean_time'])    # Number of time records

    """

    # TODO: Make grd optional
    # Only use of grd is to look work on subdomain,
    #   alternatively: use subgrid specification
    #   make attribute grd.subgrid

    N = L // 2
    assert 2 * N == L, "Only even averaging periods allowed (presently)"

    # Dimension and staggering
    if var_name == "u":  # 3D u-point
        I, J = grd.Iu, grd.Ju
        s = (slice(None), grd.Ju, grd.Iu)
    elif var_name == "v":  # 3D v-point
        I, J = grd.Iv, grd.Jv
        s = (slice(None), grd.Jv, grd.Iv)
    elif var_name == "ocean_time":  # scalar
        s = ()
    else:  # default = 3D rho-point
        I, J = grd.I, grd.J
        s = (slice(None), grd.J, grd.I)

    # First average
    MF = fid.variables[var_name][(0,) + s] / (4 * N)
    for t in range(1, 2 * N):
        MF += fid.variables[var_name][(t,) + s] / (2 * N)
    MF += fid.variables[var_name][(2 * N,) + s] / (4 * N)
    yield MF

    # Update the average
    for t in range(N + 1, n_rec - N):
        MF += fid.variables[var_name][(t + N,) + s] / (4 * N)
        MF += fid.variables[var_name][(t + N - 1,) + s] / (4 * N)
        MF -= fid.variables[var_name][(t - N,) + s] / (4 * N)
        MF -= fid.variables[var_name][(t - N - 1,) + s] / (4 * N)
        yield MF

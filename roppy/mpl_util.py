# -*- coding: utf-8 -*-


"""A set of utility functions for matplotlib

Function overview
-----------------

:func:`landmask`
  Add a land mask to plott

:func:`LevelColormap`
  Make a colormap for a sequence of levels

:func:`levelmap`
  Make a colormap for a sequence of levels

"""

# -----------------------------------
# mpl_util.py
#
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# 2010-01-05
# -----------------------------------

from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

Colormap = mpl.colors.Colormap
Colornorm = mpl.colors.Normalize

# ------------------
# Plot land mask
# ------------------


def landmask(grd, color="0.8", pcolor="pcolormesh") -> None:
    """Make a land mask, constant colour

    *grd* : An *SGrid* instance or a ROMS *mask_rho* array

    *color* : A colour description

    *pcolor* : ['pcolormesh' | 'pcolor']
                Default = 'pcolormesh'
                use 'pcolor' for savefig to eps or pdf

    Example use, mask the land green::

    >>> fid = Dataset(roms_file)
    >>> grd = SGrid(fid)
    >>> landmask(grd, 'green')
    >>> plt.plot()

    """

    # Make a constant colormap, default = grey
    constmap = plt.matplotlib.colors.ListedColormap([color])

    if isinstance(grd, np.ndarray):
        M = grd
        jmax, imax = M.shape
        X = np.arange(-0.5, imax)
        Y = np.arange(-0.5, jmax)
    else:  # grd is a roppy.SGrid instance
        M = grd.mask_rho
        X = grd.Xb
        Y = grd.Yb

    # Draw the mask by pcolor
    M = np.ma.masked_where(M > 0, M)
    if pcolor == "pcolormesh":
        plt.pcolormesh(X, Y, M, cmap=constmap)
    elif pcolor == "pcolor":
        plt.pcolor(X, Y, M, cmap=constmap)
    elif pcolor == "imshow":
        plt.imshow(X, origin="lower", cmap=constmap)


# -------------
# Colormap
# -------------

# Colormap, smlgn. med Rob Hetland
def LevelColormap(levels, cmap=None, reverse=False) -> Colormap:
    """Make a colormap based on an increasing sequence of levels

    *levels* : increasing sequence

    *cmap* : colormap, default = current colormap

    *reverse* : False|True, whether to reverse the colormap

    return value : The new colormap

    """

    # Start with an existing colormap
    if cmap is None:
        cmap = plt.get_cmap()

    # Spread the colours maximally
    nlev = len(levels)
    S = np.arange(nlev, dtype="float") / (nlev - 1)
    A = cmap(S)

    # Normalize the levels to interval [0,1]
    levels = np.array(levels, dtype="float")
    L = (levels - levels[0]) / (levels[-1] - levels[0])
    S = list(range(nlev))
    if reverse:
        levels = levels[::-1]
        L = (levels - levels[-1]) / (levels[0] - levels[-1])
        S.reverse()

    # Make the colour dictionary
    R = [(L[i], A[i, 0], A[i, 0]) for i in S]
    G = [(L[i], A[i, 1], A[i, 1]) for i in S]
    B = [(L[i], A[i, 2], A[i, 2]) for i in S]
    cdict = dict(red=tuple(R), green=tuple(G), blue=tuple(B))

    # Use
    return plt.matplotlib.colors.LinearSegmentedColormap(
        "%s_levels" % cmap.name, cdict, 256
    )


# -------------------


def levelmap(
    L, cmap=None, reverse=False, extend="neither"
) -> Tuple[Colormap, Colornorm]:
    """Make colormap and normalization from a sequence of levels

    *L* : increasing sequence of levels

    *cmap* : colormap, default = current colormap

    *reverse* : False|True, whether to reverse the colormap

    *extend* : "neither"|"min"|"max"|"both"
               handling of too small or large values

    return value : (new color map, new normalization)

    """

    N = len(L)

    # Normalize using the levels
    new_norm = mpl.colors.BoundaryNorm(L, N - 1)

    # Start with an existing colormap
    if not cmap:
        cmap = plt.get_cmap()

    # Handle the extend
    if extend == "both":
        ncol = N + 1
        c_over = True
        c_under = True
        I = slice(1, -1)
    elif extend == "min":
        ncol = N
        c_over = False
        c_under = True
        I = slice(1, None)
    elif extend == "max":
        ncol = N
        c_over = True
        c_under = False
        I = slice(None, -1)
    else:  # extend = neither
        ncol = N - 1
        c_over = False
        c_under = False
        I = slice(None)

    # Spread the colours maximally
    C = cmap(np.linspace(0.0, 1.0, ncol))
    if reverse:
        C = C[::-1]

    # Make the colormap, including under/over values
    new_map = mpl.colors.ListedColormap(C[I])
    if c_over:
        new_map.set_over(C[-1])
    if c_under:
        new_map.set_under(C[0])

    return new_map, new_norm

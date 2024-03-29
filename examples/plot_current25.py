# ------------------------------------------------------
# current.py
#
# Plot a current field at fixed depth
# Modified from the spermplot example
#
# Bjørn Ådlandsvik <bjorn@imr.no>
# 2020-03-27
# ------------------------------------------------------
# BÅ: Modified 2023-08-16,
# Comment on parameter settings for quiver
# Fix for issue 1

# -------------
# Imports
# -------------

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from roppy import SGrid
from roppy.mpl_util import landmask

# -------------------------
# User settings
# ------------------------

ncfile = "data/ocean_avg_example.nc"
timeframe = 3  # Fourth time frame

# subgrid = (1,-1,1,-1)  # whole grid except boundary cells
subgrid = (110, 170, 35, 90)

# Depth level [m]
z = 25

# --- Vector parameters ---
# These must be adjusted manually to fit the
# grid resolution, the grid size, the zoom level, ...
# plt.quiver has more arguments that can be tried.

# Distance between vectors
stride = 2
# Length scale for the arrows (larger value gives shorter arrows)
arrow_scale = 3
# Width of the arrows
arrow_width = 0.004

# Speed level (isotachs)
speedlevels = np.linspace(0, 0.5, 6)  # 0.0, 0.1, ...., 0.5

# Colormap for speed
speedcolors = "YlOrRd"

# --------------------
# Read the data
# --------------------

f = Dataset(ncfile)
grid = SGrid(f, subgrid=subgrid)

# Read 3D current for the subgrid
U0 = f.variables["u"][timeframe, :, grid.Ju, grid.Iu]
V0 = f.variables["v"][timeframe, :, grid.Jv, grid.Iv]

Mu = f.variables["mask_u"][grid.Ju, grid.Iu]
Mv = f.variables["mask_v"][grid.Jv, grid.Iv]

# ----------------------
# Handle the data
# ----------------------

# Interpolate to rho-points
U1 = 0.5 * (U0[:, :, :-1] + U0[:, :, 1:])
V1 = 0.5 * (V0[:, :-1, :] + V0[:, 1:, :])

# Interpolate to correct depth level
U = grid.zslice(U1, z)
V = grid.zslice(V1, z)

# Remove velocity at land and below bottom
U[grid.h < z] = np.nan
V[grid.h < z] = np.nan

# Compute the current speed
Speed = np.sqrt(U * U + V * V)

# Impose the stride
X = grid.X[::stride]
Y = grid.Y[::stride]
U = U[::stride, ::stride]
V = V[::stride, ::stride]

# --------------------
# Make the plot
# --------------------

# Contour plot of current speed
plt.contourf(grid.X, grid.Y, Speed, levels=speedlevels, cmap=speedcolors)
plt.colorbar()

# Make the vector plot

plt.quiver(X, Y, U, V, scale=arrow_scale, width=arrow_width)

# Plot green land mask
landmask(grid, "LightGreen")

# Set correct aspect ratio and axis limits
plt.axis("image")
plt.axis((grid.i0 + 0.5, grid.i1 - 0.5, grid.j0 + 0.5, grid.j1 - 1.5))

# Display the plot
plt.show()

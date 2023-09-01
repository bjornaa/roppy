
# ------------------------------------------------------
# spermplot.py
#
# Plot a vector field by curly vectors
#
# Preliminary version, just a small disk at the head
# instead of proper arrow head
#
# Uses subgrid facility in SGrid object
# and the trajectories module
#
# Bjørn Ådlandsvik <bjorn@imr.no>
# 2015-01-15
# ------------------------------------------------------

# -------------
# Imports
# -------------

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from roppy import SGrid
from roppy.mpl_util import landmask
from roppy.trajectories import curly_vectors

# -------------------------
# User settings
# ------------------------

ncfile = "data/ocean_avg_example.nc"
timeframe = 3  # Fourth time frame

# subgrid = (1,-1,1,-1)  # whole grid except boundary cells
subgrid = (110, 170, 35, 90)

# Distance between sperms
stride = 2

# "time"-step [s]
dt = 20000

# Number of time steps per sperm
nstep = 10

# Numerical order (1 = Euler forward, 4 = Runge-Kutta)
order = 4

# Speed level (isotachs)
# speedlevels = np.linspace(0, 1, 11)  # 0.0, 0.1, ...., 1.0
speedlevels = np.linspace(0, 0.5, 6)  # 0.0, 0.1, ...., 0.5

# Colormap for speed
speedcolors = "YlOrRd"

# Color of spermplot
spermcolor = "black"
# spermcolor = 'blue'
# spermcolor = 'white'  # Use white if speedcolors are dark


# Length is scaled by dt*nstep
# order=4 => nstep can be smaller => faster
# Reasonable length depend on grid resolution and grid shape
# Reasonable values for stride depend on subgrid shape
# TODO: Find reasonable defaults, for grids and subgrids

# --------------------
# Read the data
# --------------------

f = Dataset(ncfile)
grid = SGrid(f, subgrid=subgrid)

# Read surface current for the subgrid
U = f.variables["u"][timeframe, -1, grid.Ju, grid.Iu]
V = f.variables["v"][timeframe, -1, grid.Jv, grid.Iv]

Mu = f.variables["mask_u"][grid.Ju, grid.Iu]
Mv = f.variables["mask_v"][grid.Jv, grid.Iv]

# f.close()

# ----------------------
# Handle the data
# ----------------------

# Make sure velocity = 0 on land
U = Mu * U
V = Mv * V

# Compute the curly vectors
X, Y = curly_vectors(grid, U, V, stride=stride, nstep=nstep, dt=dt, order=order)

# Compute the current speed
U1 = 0.5 * (U[:, :-1] + U[:, 1:])
V1 = 0.5 * (V[:-1, :] + V[1:, :])
Speed = np.sqrt(U1 * U1 + V1 * V1)

# --------------------
# Make the plot
# --------------------

# Contour plot of current speed
plt.contourf(grid.X, grid.Y, Speed, levels=speedlevels, cmap=speedcolors)
plt.colorbar()

# Make the spermplot
plt.plot(X, Y, color=spermcolor)
plt.plot(X[-1, :], Y[-1, :], linestyle="", marker=".", color=spermcolor)


# Plot green land mask
landmask(grid, "green")

# Set correct aspect ratio and axis limits
plt.axis("image")
plt.axis((grid.i0 + 0.5, grid.i1 - 1.5, grid.j0 + 0.5, grid.j1 - 1.5))

# Display the plot
plt.show()

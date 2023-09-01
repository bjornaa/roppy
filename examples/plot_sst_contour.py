#!/usr/bin/env python

"""Read SST data and make a simple contour plot in matplotlib"""

# --------------------------------
# ROMS python demo
# Bjørn Ådlandsvik
# 2007-06-19
# --------------------------------

# --------
# Imports
# --------

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

# --------------
# User settings
# --------------

romsfile = "./data/ocean_avg_example.nc"
tstep = 3  # 4-th time frame

# -------------------
# Extract the data
# -------------------

fid = Dataset(romsfile)
sst = fid.variables["temp"][tstep, -1, :, :]
M = fid.variables["mask_rho"][:, :]
fid.close()

# --------------
# Data handling
# --------------

# Mask out the SST data with NaN on land
# to improve the contour plot
sst[M < 1] = np.NaN

# -------------
# Plot
# -------------

# Contour plot with colorbar
plt.contourf(sst)
plt.colorbar()

# Plot with correct aspect ratio
plt.axis("image")
plt.show()

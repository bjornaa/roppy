#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Read SST data and make a simple pcolor plot in matplotlib"""

# --------------------------------
# ROMS python demo
# Bjørn Ådlandsvik
# 2010-10-04
# --------------------------------

# --------
# Imports
# --------

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from roppy import SGrid

# --------------
# User settings
# --------------

romsfile = "./data/ocean_avg_example.nc"
tstep = 3  # 4-th time frame

# -------------------
# Extract the data
# -------------------

fid = Dataset(romsfile)
grd = SGrid(fid)
sst = fid.variables["temp"][tstep, -1, :, :]
# fid.close()

# --------------
# Data handling
# --------------

# Make a masked array of the SST data
# to improve the pcolor plot
sst = np.ma.masked_where(grd.mask_rho < 1, sst)

# -------------
# Plot
# -------------

# Use X = grd.Xb, Y = grd.Yb
# to make matplotlib user coordinates
# equal to the ROMS grid coordinates

# pcolormesh is faster than pcolor

plt.pcolormesh(grd.Xb, grd.Yb, sst)
plt.colorbar()

# Plot with correct aspect ratio
plt.axis("image")
plt.show()

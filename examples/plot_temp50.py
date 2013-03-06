# -*- coding: utf-8 -*-

# --------------------------------------------------
# Plot temperature at 50 m, using the SGrid class
#
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
# Created: 2010-01-20
# --------------------------------------------------

# ---------
# Imports
# ---------

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from roppy import SGrid
from roppy.mpl_util import landmask

# -----------------
# User settings
# -----------------

roms_file = 'data/ocean_avg_example.nc'
var = 'temp'                  # name of variable in NetCDF file
tstep  = 3                    # 4th time step 
depth  = 50                   # plot depth

# --------------------
# Extract the data
# --------------------

fid = Dataset(roms_file)
grd = SGrid(fid)

# Read the 3D temperature field
F = fid.variables[var][tstep, :, :, :]
long_name = fid.variables[var].long_name

fid.close()

# ------------------
# Handle the data
# ------------------

# Interpolate temperature to the depth wanted
F = grd.zslice(F, depth)

# Mask away temperatures below bottom
F = np.ma.masked_where(grd.h < abs(depth), F)  # numpy masked array
#F[grd.h < depth] = np.nan 


# ----------
# Plotting
# ----------

# Make a filled contour plot of the temperature values
plt.contourf(F)

# A slightly nicer colorbar matching the black isolines
plt.colorbar(drawedges=1)

# Draw black contour lines at the same isolevels
plt.contour(F, colors='black')

# Plot the land mask in grey
# Use keyword pcolor='pcolor' for savefig to eps or pdf
landmask(grd)
#landmask(grd, pcolor='pcolor')


# Fix aspect ratio, so that grid cells  are squares
plt.axis('image')

# Display the plot
plt.show()
#plt.savefig('a.pdf')


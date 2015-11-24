import numpy as np
from netCDF4 import Dataset

# Import development version of roppy
import sys
sys.path = ['..'] + sys.path
import roppy

# --- EDIT -----------------

# ROMS file
romsfile = 'data/ocean_avg_example.nc'

# Section definition
lon0, lat0 = -0.67, 60.75  # Shetland
lon1, lat1 =  4.72, 60.75  # Feie

# --- EDIT ------------------

# Make a grid object
f = Dataset(romsfile)
grd = roppy.SGrid(f)

# Get grid coordinates of end points
x0, y0 = grd.ll2xy(lon0, lat0)
x1, y1 = grd.ll2xy(lon1, lat1)
# Find nearest rho-points
i0, j0, i1, j1 = [int(round(v)) for v in x0, y0, x1, y1]

# Make a Section object
sec = roppy.linear_section(i0, i1, j0, j1, grd)

# Read in a 3D temperature field
temp = f.variables['temp'][0,:,:,:]

# Interpolate to the section
temp_sec = sec.sample3D(temp)

# Compute mean temperature along section
# using trapezoidal integration

print "mean tempeature = ", np.sum(sec.Area * temp_sec) / np.sum(sec.Area)

# TODO: Make a mean method in the Section class
#   Usage: sec.mean(temp_sec)
#   or even directly from 3D: sec.mean(temp)

"""Plot the bottom topography for the whole domain"""

# ----------------------------------
# plot_topo.py
#
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# 2010-10-04
# ----------------------------------

# ---------------
# Imports
# ---------------

import matplotlib.pyplot as plt
from netCDF4 import Dataset
from roppy import SGrid, landmask, levelmap

# -----------------
# User settings
# -----------------

# File with bottom topography
roms_file = "data/ocean_avg_example.nc"

# Isolevels [m]
L = [25, 50, 100, 250, 500, 1000, 2500]

# ------------------
# Extract the data
# ------------------

fid = Dataset(roms_file)
grd = SGrid(fid)

# ---------------
# Plot the data
# ---------------

# Plot bathymetry as filled contours
# with colours equally distributed from the default colour map
# extending above max and below min level
extend = "both"
cmap, norm = levelmap(L, extend=extend, reverse=True)
plt.contourf(grd.h, levels=L, cmap=cmap, norm=norm, extend=extend)
plt.colorbar()

# Add black contour lines
plt.contour(grd.h, levels=L, colors="black")

# Add a grey land mask
landmask(grd, color="grey")

plt.axis("image")
plt.show()

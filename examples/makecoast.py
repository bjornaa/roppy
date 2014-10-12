#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract a closed coast line

Extracts a coast line from GSHHS using the
advanced polygon handling features in Basemap

The polygons are saved to a two-columns
text file, using Nans to sepatate the polygons.
An example of how to read back the data and
plot filled land is given in pcoast.py

"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn at imr.no>
# Institute of Marine Research
# 2014-10-12
# ----------------------------------

# ---------------
# Imports
# ---------------

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# -----------------
# User settings
# -----------------

# Geographical extent (should include all land in domain)
lon0, lon1 = -12, 16      # Longitude range
lat0, lat1 = 47, 66       # Latitude range

# Choose GSHHS resolution
res = 'i'  # intermediate resolution

# Output coast file
outfile = 'data/coast.dat'

# ------------------------------
# Set up Basemap map projection
# ------------------------------

# Use cylindrical equidistand projection
# i.e. x = lon, y = lat
m = Basemap(projection = 'cyl',
            llcrnrlon  = lon0,
            llcrnrlat  = lat0,
            urcrnrlon  = lon1,
            urcrnrlat  = lat1,
            resolution = res)

# ----------------------------
# Get the coast polygon data
# ----------------------------

polygons = []
for i, p in enumerate(m.coastpolygons):
    # Use only coast polygons (ignore lakes)
    if m.coastpolygontypes[i] == 1:
        polygons.append(p)

# --------------------
# Save the coast data
# --------------------

with open(outfile, 'w') as fid:
    for p in polygons:                  # Loop over the polygons
        for v in zip(*p):               # Loop over the vertices
            fid.write('{:7.3f}{:7.3f}\n'.format(*v))
        fid.write('    Nan    Nan\n')  # Separate the polygons

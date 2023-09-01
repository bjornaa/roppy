"""Extract a closed coast line

Extracts a coast line from GSHHS using the
advanced polygon handling features in cartopy and shapely

The polygons are saved to a two-columns
text file, using Nans to sepatate the polygons.
An example of how to read back the data and
plot filled land is given in pcoast.py

"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn at imr.no>
# Institute of Marine Research
# 2014-10-12
# 2021-12-01   # Moved from basemap to cartopy
# ----------------------------------

# ---------------
# Imports
# ---------------

import cartopy.io.shapereader as shapereader
from shapely import geometry

# -----------------
# User settings
# -----------------

# Geographical extent (should include all land in domain)
lon0, lon1 = -12, 16  # Longitude range
lat0, lat1 = 47, 66  # Latitude range

# Choose GSHHS resolution
res = "i"  # intermediate resolution
min_area = 0.01  # Minimum area of polygon to include

# Output coast file
outfile = "data/coast.dat"

# ---------------------------

# Global coastline from GSHHS as shapely collection generator
path = shapereader.gshhs(scale=res)
coast = shapereader.Reader(path).geometries()

# Restrict the coastline to the regional domain
bbox = geometry.box(lon0, lat0, lon1, lat1)
coast0 = (bbox.intersection(p) for p in coast if bbox.intersects(p))

# Make a list of the polygons in the intersection
coast = []
for p in coast0:
    if isinstance(p, geometry.Polygon):
        coast.append(p)
    elif isinstance(p, geometry.MultiPolygon):
        coast.extend(p.geoms)

# Filter out very small islands
coast = [p for p in coast if p.area >= min_area]

# --------------------
# Save the coast data
# --------------------

with open(outfile, mode="w") as fid:
    for p in coast:  # Loop over the polygons
        for v in zip(*p.boundary.xy):  # Loop over the vertices
            fid.write("{:7.3f}{:7.3f}\n".format(*v))
        fid.write("    Nan    Nan\n")  # Separate the polygons

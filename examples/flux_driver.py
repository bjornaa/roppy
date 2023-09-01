"""flux_driver.py

Compute time series of volume flux through selected sections
from a sequence of ROMS history/average files
"""

# -------------------------------------
# Driver for flux calculations
#
# Bjørn Ådlandsvik <bjorn at imr.no>
# Institute of Marine Research
# ------------------------------------

from pathlib import Path

import numpy as np
from netCDF4 import Dataset, MFDataset

from roppy import FluxSection, SGrid, staircase_from_line

# End points of sections in longitude/latitude
sec_dict = {
    "Feie-Shetland": {"lat0": 60.750, "lon0": 4.617, "lat1": 60.750, "lon1": -0.667},
    "Utsira-W": {"lat0": 59.283, "lon0": 5.033, "lat1": 59.283, "lon1": -2.223},
    "Torungen-Hirtshals": {
        "lat0": 58.400,
        "lon0": 8.767,
        "lat1": 57.633,
        "lon1": 9.867,
    },
}

# Sections to use
sections = ["Feie-Shetland", "Utsira-W"]

datadir = Path("./data")

gridfile = datadir / "ocean_avg_example.nc"

datafile_format = "ocean_avg_*.nc"

outfile = "flux.dat"


# ----------------------------------------------

# Make SGrid object

f0 = Dataset(gridfile)
grd = SGrid(f0)

# Make FluxSection objects

for secname in sections:
    print(secname)
    sec = sec_dict[secname]
    # End points in grid coordinates
    x0, y0 = grd.ll2xy(sec["lon0"], sec["lat0"])
    x1, y1 = grd.ll2xy(sec["lon1"], sec["lat1"])
    # Closest psi-points (ROMS indices)
    i0, i1, j0, j1 = (int(np.ceil(v)) for v in (x0, x1, y0, y1))
    # Staircase
    I, J = staircase_from_line(i0, i1, j0, j1)
    # FluxSection
    sec["fsec"] = FluxSection(grd, I, J)
    sec["netflux"] = []
    sec["posflux"] = []

# Generate file list
datafiles = list(datadir.glob(datafile_format))
datafiles.sort()
print("Data files : ", datafiles)

f = MFDataset(datafiles)
# ntimes = len(f.dimensions['ocean_time'])
ntimes = len(f.dimensions["time"])
for t in range(ntimes):
    U = f.variables["u"][t, :, :, :]
    V = f.variables["v"][t, :, :, :]
    for secname in sections:
        sec = sec_dict[secname]
        F0, F1 = sec["fsec"].transport(U, V)
        sec["netflux"].append(F0 * 1e-6)
        sec["posflux"].append(F1 * 1e-6)

# Save the dataset
with open(outfile, mode="w") as f1:
    # First header line
    nsec = len(sections)
    outstring = (nsec * "{:^20s}").format(*sections)
    f1.write(outstring + "\n")

    # Second header line
    outstring = nsec * ("       net  positive")
    f1.write(outstring + "\n")

    # Time series
    for t in range(ntimes):
        netflux = [sec_dict[secname]["netflux"][t] for secname in sections]
        posflux = [sec_dict[secname]["posflux"][t] for secname in sections]
        outstring = "".join(
            [f"{nf:10.3f}{pf:10.3f}" for (nf, pf) in zip(netflux, posflux)]
        )
        f1.write(outstring + "\n")

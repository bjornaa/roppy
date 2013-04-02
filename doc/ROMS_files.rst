==============================
ROMS files and the SGrid class
==============================

.. Author: Bjørn Ådlandsvik
.. Address: Institute of Marine Research, Bergen, Norway
.. e-mail: bjorn@imr.no
.. date: 2010-10-30

ROMS output files
=================

Output files from ROMS are instances of NetCDF files [link].
The format is netcdf3-classic?, netcdf3-64bit
If ROMS is linked with the netcdf4-library the format is
netcdf4-classic. [Sjekk opp i dette, nyeste versjoner kan
være endret].

ROMS files since version xxx follows the CF-standard [link].
[sjekk at dette er sant]

There basic output files are history files containing snapshots
of the model state and averages files containing time averages.
In addition there can be station and float files and more?

The history and average files are self-contained containing the
necessary information on the spatial and temporal discretization,
some other settings [sjekk]. 

ROMS files may be further prosessed, for instance to compute
monthly averages or restrict the results to a subgrid.
Such secondary files should ideally follow ROMS' auto-documentation
practise, but often much of the information is left out.

Variabel-navn, ocean_time


Python and NetCDF
=================
 
There exist several options for handling NetCDF files with python.
The first was the NetCDFFile class contained in the scientific-python
package developed by Konrad Hinsen. Most of the later packages follows
the conventions layed out in this class, making it easy to switch
between different packages. An interesting option is pypunere, this
is a pure python package for NetCDF3 that does not depend any NetCDF
library, making the installation trivial. It can easily be used
on external machines where you do not have control over the software
installed centrally.

In the examples here the netcdf4-python package [link] by Geoffrey
Whittaker is used. This is the recommended solution as it undergoes
active development, supports both netCDF-3 and netCDF-4, and has some
extra gooddies. In particular the multi-file class MFDataset goes hand
in hand with ROMS splitting of output files. The downside is that it
is not (yet) part of standard Linux distributions and require somewhat
complicated manual installation. The installation instructions are,
however, very good.

[Hvordan bruke pupynere in the examples]

Indexing
========

ROMS is written in fortran and has flexible indexing. The rho-points
are zero-based, going from 0 to Lm+1 (with interior cells from 1 to
Lm) in x (xsi) direction in 0 to Mm+1 in y (eta) direction. Similalry
the vertical s-interfaces are indexed from 0 at bottom (s=-1) to
N at surface (s=0) while the s-levels are indexed from 1 to N.

In python all indexing is zero based. This works nicely with the
rho-points but some care must be taken for the velocity points.
For instance the velocity points with x=5.5 (between rho-points 5 and
6) have x-index 6 in ROMS and 5 in python.

Ta inn figur

The ordering of the indices must also be considered. ROMS uses the
ordering (x, y, depth, time) or more precisely (xsi, eta, s, ocean_time).
Python follows the C-style of multiple indices, making the fields
come out with indices in reversed order (time, depth, y, x). This is
backwards as first, but works well with visualisation in matplotlib.

As an example to get the surface temperature in grid cell x=100, y=50
at the 5-th time frame in the file. In ROMS fortran notation 
temp(100,50,N,5). In python this is read by::

  fid = Dataset('ocean_his.nc')
  value = fid.variables['temp'][4, -1, 50, 100]

Here the time frames are indexed from zero, giving time index 4 for
the 5th frame. The -1 is a convenient python notation for the last
index, here N-1 giving the surface value.



The SGrid class
===============

A python class, SGrid, has been designed to simplify working with ROMS
output files. See [link] for full documentation. The name is choosen
to distinguish it from other useful classes like octant's grid class
and soda2romsi ....  Initially the S could stand for simple, but I
think of it as a reminder that it is a 3D object, containing the info
needed for the vertical s-coordinates.

It is initiated by an opened ROMS NetCDF file. ROMS history and average files
contain the necessry meta-information. It is considered a bug if
SGgrid barks at one of these files. Curiously, it could not be initiated
by a ROMS grid file as this file type does not contain information on
the vertical discretization. This has been improved by allowing the
vertical information to be passed on as a dictionary [eller en egen
klasse: fordel kan bruke en åpen fil]. 
Similarly, a typical ini-file can not be used as it does not contain
enough information on the horizontal discretization.

A typical sequence may look something like::

  from netCDF4 import Dataset
  import roppy
  ...
  fid = Dataset('ocean_his.nc')
  grd = roppy.SGrid(fid)
  ...

The attributes of an SGrid object follows the variable names in the
ROMS file. For instance the land-sea mask is available as
grd.mask_rho. The vertical structure is available as the 3D array
grd.z_rho (and grd.z_w at the s-coordinate interfaces) following the
internally used ROMS naming convention. [sjekk]

As computers gets more powerful, the number of grid cells that can be
used grows. However, to analyze or simply to plot 3000 x 1000 grid
cells on a normal computer display is not ...  One of the features of
the SGrid object is to make it simple to work with subgrids. For
instance, SGrid knows which slices to use to access rho-points and
velocity points, this is tedious and error-prone when done manually.




  


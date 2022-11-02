========================
Horizontal scalar fields
========================

.. Author: Bjørn Ådlandsvik
.. Address: Institute of Marine Research, Bergen, Norway
.. e-mail: bjorn@imr.no
.. date: 2010-11-01


Disposisjon:
Ta verktøyene først, eller start med maskering m.m.
Kan evt. starte med enkelt felt som topografi
Kan også gjøre som tutorial, ta SST og bygg opp.
Viss logikk, ta analyse/data først deretter plotting

Reading 2D fields
-----------------

Some fields such as topography and sea surface elevation are
naturally 2D. Assessing a whole 2D field from ROMS is esali done

::

  fid.variables['h'][:,:]         # Topography
  fid.variables['h'][:]           # Short hand for the whole h array
  grd.h                           # Topography available from SGrid
  fid.variables['zeta'][t,:,:]    # SSH at time frame t
  fid.variables['temp'][t,-1,:,:] # SST at time frame t
  fid.variables['u'][t,0,:,:]     # Bottom current in x-direction at time t
  fid.vatiables['vbar'][t,:,:]    # Depth averaged current in y-direction

Subfields
---------

To extract a field on a subgrid, slightly more advanced slicing is
used. A subgrid can be spesified by limiters i0, i1 in the
x-direction and j0, j1 in the y-directions. As usual in python, the
upper end points are not included, the subggrid is given by 0 <= i <
i1 and j0 <= j < j1, a total of (i1-i0)x(j1-j0) grid cells.

Working with a subgrid is simple with the SGrid class. Assuming a
subgrid definition::

  grd = SGrid(fid, subgrid=(i0,i1,j0,j1))

The grid variables are then restricted to the subgrid making the
following statements equivalent::

  fid.variables['h'][j0:j1,i0:i1]   
  grd.h

For data variables the grid class offers handy slice objects,
making the following equal::

  fid.variables['temp'][t, -1, j0:j1, i0:i1]
  fid.variables['temp'][t, -1, grd.J, grd.I]

Velocitiy fields on a subgrid require some care with the indexing.
SGrid keeps track of this, also in the boundary cases where
i0 = 0 or i1 = None (shorthand for Lm+1)::

  fid.variables['u'][t, -1, j0:j1, i0-1:i1]  # 1 <= i0 < i1 <= Lm
  fid.variables['u'][t, -1, grd.Ju, grd.Iu]
 
  fid.variables['u'][t, -1, j0:j1, i0-1:]    # 1 <= i0 < i1 = Lm+1
  fid.variables['u'][t, -1, grd.Ju, grd.Iu]


[Ta restriksjon av vektor-felt spesielt med grensetilfeller]


Horizontal slicing
------------------

A common way to make 2D fields for analysis and plotting is by making
a horizontal slice of a 3D field. The simplest case is slices along
the s-levels, which can be done directly by the netCDF object. To
extract the k-th s-level do::
 
  temp_k = fid.variables['temp'][t, k, :, :]

[Flytte dette før]

More useful is to slice to a fixed depth, or more generally to a
prescribed horizontally varying depth level. SGrid does this by linear
interpolation in the vertical. This may be improved later by also
offering ROMS own parabolic spline interpolation.

Using SGrid the temperature at 50 meter depth is given as::

  temp3D = fid.variables['temp'][t, :, :, :] # 3D temperature field
  temp50 = grd.zslice(temp3D, 50)  # sign of second argument is 

[sjekk at roppy gjør dette, hvis ikke fiks det]

[**Flytt til kapittel om utilities**]
The romsutil toolbox in roppy offers low-level utilities for
performing this task::

  temp3D = fid.variables['temp'][tstep, :, :, :] # 3D temperature field
  C = fid.variables['Cs_r'][:]              # s-coord stretching
  H = fid.variables['h'][:,:]               # Bottom topography 
  Hc = fid.variables['hc'].getValue()

This can be used to make a 3D array *z_rho* of the vertical strukture,
where *z_rho[k,j,i]* is the depth of the k-th rho-point in grid cell
(i,j)::

  z_rho = roppy.sdepth(H, Hc, C)

Using *z_rho* linear interpolation in the vertical is simply done by
*zslice* where the depth parameter should be negative::

  temp2D = roppy.zslice(temp3D, z_rho, -abs(depth))



More advanced horizontal slicing
--------------------------------

The low-level module offer more advanced slicing such as averaging
between two depth levels (possibly varying in space).

[beskriv, test, generaliser til Sgrid]

A multi-slice routine is also available, which is faster than looping
over a sequence of depth levels.



Masking
-------

For analysis or plotting it is useful to mask out undefined values,
for instance at land or below bottom. Numpy
offers two ways of masking an array, *masked arrays* and *nan*.  The
masked array is the traditional way and works with most matplotlib
functions, such as *contour*, *contourf*, *pcolor*, *pcolormesh*, and
*imshow*. The masking is done by

To mask out land values from a 2D horizontal field do::

  >>> F = np.ma.masked_where(grd.mask_rho < 1, F) 

To mask out 50 m values below botton

  >>> temp50 = grd.zslice(temp3D, 50)  # sign of second argument is 
  >>> temp50 = np.ma.masked_where(grd.h > 50, temp50)
  
[Mulig å få dette inn i zslice-metoden? f.eks

  >>> temp50 = grd.zslice(temp3D, 50, bottom_mask=True)


The alternative, with *nan* is similar to Matlab. Unfortunately
in the present version of matplotlib (0.99.1) this does not
work with *pcolor* and *pcolormesh*. Land masking can be simply done by::

  >>> F[grd.mask_rho < 1] = np.nan



Plotting 2D scalar fields
=========================

Matplotlib offer a series of tools to visualize 2D scalar fields,
including contour, contourf, pcolor, pcolormesh, and imshow.

In the examples below sea surface temperature at time frame 5
(python index 4) from a ROMS file
$ROPPY/examples/data/ocean_avg_0014.nc
is used::

   fid = Dataset('../examples/data/ocean_avg_0014.nc')
   sst = fid.variables['temp'][4, -1, :, :]
  

Contouring
----------

Contouring is the traditional way of presenting a 2D scalar field.
The *contour* function plots contour lines, and *contourf* fills
colours between the contour levels. 

By default *contour* gives different colours to the contours, which
can identified by a *colorbar* call::

  plt.pcolor(sst)
  plt.colorbar()

Using *clabel* matplotlib does a descent job in labelling the contours.
::

  c = plt.pcolor(sst, levels=range(7,15), colors='black')
  plt.clabel(c, ...)
  




pcolor and pcolormesh
---------------------

These perfo

contour
-------




Contour plots of horizontal fields are a very common way to present
model results.



Masking
=======

Subgrid
=======

Horizontal slices
=================

Land mask
=========



Topography and levelmap
----------------------------

The simple example above showed how to plot a 2D field such as
SST. The same procedure can be used for the bottom topography.
To show details on the shelf it is useful to select a non-uniform
sequence of iso-levels::

  L = [10,25,50,100,250,500,1000,2500,5000]

However, using a simple *contourf* function with these levels still
hides all shallow details in one end of the colour map.
The `mpl_util` module has a function *levelmap* to fix this.
The call::

  levelmap(L)

returns a colormap and a normalization spreading the whole colour spectrum
evenly on the choosen levels. An keyword argument can be used to reverse the
colormap, for instance to have the deepest ocean in blue. The contour plot is
then done by::

  cmap, norm = roppy.levelmap(L, extend="both", reverse=True)
  plt.contourf(grd.h, levels=L, cmap=cmap, norm=norm, extend="both")

The final plot looks like this

.. plot:: ../examples/plot_topo.py


  



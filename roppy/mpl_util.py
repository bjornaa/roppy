# -*- coding: utf-8 -*-


"""A set of utility functions for matplotlib

Function overview
-----------------

:func:`landmask`
  Add a land mask to plott

:func:`LevelColormap`
  Make a colormap for a sequence of levels

"""

# -----------------------------------
# mpl_util.py
#
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# 2010-01-05
# -----------------------------------

import numpy as np
import matplotlib.pyplot as plt


# ------------------
# Plot land mask
# ------------------

def landmask(grd, color='0.8', pcolor='pcolormesh'):
   """Make a land mask, constant colour

   *grd* : An *SGrid* instance or a ROMS *mask_rho* array

   *color* : A colour description

   *pcolor* : ['pcolormesh' | 'pcolor']
               Default = 'pcolormesh'
               use 'pcolor' for savefig to eps or pdf
     

   Example use, mask the land green::

   >>> fid = Dataset(roms_file)
   >>> grd = SGrid(fid)
   >>> landmask(grd, 'green')
   >>> plt.plot()

   """

   # Make a constant colormap, default = grey
   constmap = plt.matplotlib.colors.ListedColormap([color])
   
   try:  # Try SGrid object first
      M = grd.mask_rho
      X = grd.Xb
      Y = grd.Yb
   except AttributeError:  # Otherwise M is a mask
      M = grd
      jmax, imax = M.shape
      X = -0.5 + np.arange(imax+1)                    
      Y = -0.5 + np.arange(jmax+1)  
      
   # Draw the mask by pcolor
   M = np.ma.masked_where(M > 0, M)
   if pcolor == "pcolormesh":
      plt.pcolormesh(X, Y, M, cmap=constmap)
   elif pcolor == "pcolor":
      plt.pcolor(X, Y, M, cmap=constmap)
   elif pcolor == "imshow":
      plt.imshow(X, origin='lower', cmap=constmap)
      

# -------------
# Colormap
# -------------

# Colormap, smlgn. med Rob Hetland
def LevelColormap(levels, cmap=None, reverse=False):
    """Make a colormap based on an increasing sequence of levels

    *levels* : increasing sequence
    
    *cmap* : colormap, default = current colormap

    *reverse* : False|True, whether to reverse the colormap

    return value : The new colormap

    """
    
    # Start with an existing colormap
    if cmap == None:
        cmap = plt.get_cmap()

    # Spread the colours maximally
    nlev = len(levels)
    S = np.arange(nlev, dtype='float')/(nlev-1)
    A = cmap(S)

    # Normalize the levels to interval [0,1]
    levels = np.array(levels, dtype='float')
    L = (levels-levels[0])/(levels[-1]-levels[0])
    S = range(nlev)
    if reverse:
       levels = levels[::-1]
       L = (levels-levels[-1])/(levels[0]-levels[-1])
       S.reverse()

    # Make the colour dictionary
    R = [(L[i], A[i,0], A[i,0]) for i in S]
    G = [(L[i], A[i,1], A[i,1]) for i in S]
    B = [(L[i], A[i,2], A[i,2]) for i in S]
    cdict = dict(red=tuple(R),green=tuple(G),blue=tuple(B))

    # Use 
    return plt.matplotlib.colors.LinearSegmentedColormap(
        '%s_levels' % cmap.name, cdict, 256)



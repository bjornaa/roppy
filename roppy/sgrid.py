# -*- coding: utf-8 -*-
"""

Classes
-------

:class:`SGrid`
  Simple grid class
:class:`Section`
  Class for vertical sections

"""

# -----------------------------------
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
# Bergen, Norway
# 2010-09-30
# -----------------------------------


import numpy as np
from netCDF4 import Dataset

from depth import sdepth, zslice, s_stretch
from sample import sample2D, bilin_inv

# ------------------------------------------------------
# Classes
# ------------------------------------------------------

class SGrid(object):

    """Simple ROMS grid object

    Simple, minimal ROMS 3D grid object, for keeping important
    information together. Meant to be compatible with
    roms2soda's grdClass.

    Note: Can not (yet) be initialized from a standard grd-file,
    use initial, history or average file or supply extra vertical
    information by Vinfo or Vfile.

    Typical usage::

    >>> fid = Dataset(roms_file)
    >>> grd = SGrid(fid)

    More arguments::

    >>> fid = Dataset(roms_file)
    >>> Vinfo = {'N' : 32, 'hc' : 10, 'theta_s' : 0.8, 'theta_b' : 0.4}
    >>> grd = SGrid(fid, subgrid=(100, 121, 60, 161), Vinfo=Vinfo)

    """

    def __init__(self, ncid, subgrid=None, Vinfo=None, Vfile=None):

        # ----------------------------------
        # Handle the vertical discretization
        # ----------------------------------

        if Vinfo:
            self.N = Vinfo['N']
            self.hc = Vinfo['hc']
            # Trengs ikke utenfor her
            if Vinfo.has_key('Vstretching'):
                self.Vstretching = Vinfo['Vstretching']
            else:
                self.Vstretching = 1
            if Vinfo.has_key('Vtransform'):  # Denne trenger self
                self.Vtransform = Vinfo['Vtransform']
            else:
                self.Vtransform = 1

            self.Cs_r = s_stretch(self.N, Vinfo['theta_s'], Vinfo['theta_b'],
                                  stagger='rho', Vstretching=self.Vstretching)
            self.Cs_w = s_stretch(self.N, Vinfo['theta_s'], Vinfo['theta_b'],
                                  stagger='w', Vstretching=self.Vstretching)

        elif Vfile:  # Read vertical info from separate file

            f0 = Dataset(Vfile)

            self.hc = f0.variables['hc'].getValue()
            self.Cs_r = f0.variables['Cs_r'][:]
            self.Cs_w = f0.variables['Cs_w'][:]

            # Vertical grid size
            self.N = len(self.Cs_r)

            # Vertical transform
            self.Vtransform = 1  # Default
            try:   # Look for standard_name attribute of variable s_rho
                v = f0.variables['s_rho']
                if v.standard_name[-1] == '2':
                    self.Vtransform = 2
            # No variable s_rho or no standard_name attribute
            except (KeyError, RuntimeError):
                pass                    # keep old default Vtransform = 1

            f0.close()

        else:  # Read vertical info from the file

            self.hc = ncid.variables['hc'].getValue()
            self.Cs_r = ncid.variables['Cs_r'][:]
            self.Cs_w = ncid.variables['Cs_w'][:]

            # Vertical grid size
            self.N = len(self.Cs_r)

            # Vertical transform
            self.Vtransform = 1  # Default
            try:   # Look for standard_name attribute of variable s_rho
                v = ncid.variables['s_rho']
                if v.standard_name[-1] == '2':
                    self.Vtransform = 2

            # No variable s_rho or no standard_name attribute
            except (KeyError, RuntimeError):
                pass                    # keep default Vtransform = 1

        # ---------------------
        # Subgrid specification
        # ---------------------

        Mp, Lp = ncid.variables['h'].shape
        if subgrid:
            i0 = subgrid[0]
            i1 = subgrid[1]
            j0 = subgrid[2]
            j1 = subgrid[3]
            if i0 < 0: i0 += Lp
            if i1 < 0: i1 += Lp
            if j0 < 0: j0 += Mp
            if j1 < 0: j1 += Mp
            # should have test 0 <= i0 < i1 = Lp
            # should have test 0 <= j0 < j1 = Mp
            #self.Lp = self.i1 - self.i0
            #elf.Mp = self.j1 - self.j0
            self.i0, self.i1 = i0, i1
            self.j0, self.j1 = j0, j1
        else:
            self.i0, self.i1 = 0, Lp
            self.j0, self.j1 = 0, Mp

        # Shape
        self.shape = (self.j1-self.j0, self.i1-self.i0)

        # Slices
        self.I = slice(self.i0, self.i1)
        self.J = slice(self.j0, self.j1)

        # U and V-points
        i0_u = max(0, self.i0-1)
        i1_u = min(self.i1, Lp-1)
        j0_v = max(0, self.j0-1)
        j1_v = min(self.j1, Mp-1)
        self.i0_u = i0_u
        self.j0_v = j0_v

        self.Iu = slice(i0_u, i1_u)
        self.Ju = self.J
        self.Iv = self.I
        self.Jv = slice(j0_v, j1_v)

        # ---------------
        # Coordinates
        # ---------------

        # Limits
        self.xmin = float(self.i0)
        self.xmax = float(self.i1 - 1)
        self.ymin = float(self.j0)
        self.ymax = float(self.j1 - 1)

        # Grid cell centers
        self.X = np.arange(self.i0, self.i1)
        self.Y = np.arange(self.j0, self.j1)
        # U points
        #self.Xu = np.arange(self.i0_u, self.i1_u)
        #self.Yu = self.Y
        # V points
        #self.Xv = self.X
        #self.Yv = np.arange(self.j0_v, self.j1_v)
        # Grid cell boundaries = psi-points
        self.Xb = np.arange(self.i0-0.5, self.i1)
        self.Yb = np.arange(self.j0-0.5, self.j1)

        # Read variables from the NetCDF file
        # ------------------------------------
        
        self.h = ncid.variables['h'][self.J, self.I]
        # mask_rho should not be masked
        self.mask_rho = np.array(ncid.variables['mask_rho'][self.J, self.I])

        try:
            self.pm = ncid.variables['pm'][self.J, self.I]
            self.pn = ncid.variables['pn'][self.J, self.I]
        except KeyError:
            pass
        try:
            self.lon_rho = ncid.variables['lon_rho'][self.J, self.I]
            self.lat_rho = ncid.variables['lat_rho'][self.J, self.I]
        except KeyError:
            pass
        try:
            self.angle = ncid.variables['angle'][self.J, self.I]
            self.f = ncid.variables['f'][self.J, self.I]
        except KeyError:
            pass

        # ---------------------
        # 3D depth structure
        # ---------------------

        self.z_r = sdepth(self.h, self.hc, self.Cs_r,
                          stagger='rho', Vtransform=self.Vtransform)
        self.z_w = sdepth(self.h, self.hc, self.Cs_w,
                          stagger='w', Vtransform=self.Vtransform)

# Wrappers for romsutil functions

    # Unødvendig?
    def sample2D(self, F, X, Y, mask=True, undef=np.nan):
        if mask:
            return sample2D(F, X, Y, mask=self.mask_rho,
                            undef_value=undef)
        else:
            return sample2D(F, X, Y)


    def zslice(self, F, z):
        return zslice(F, self.z_r, -abs(z))

    def xy2ll(self, x, y):
        return sample2D(self.lon_rho, x, y),    \
               sample2D(self.lat_rho, x, y)

    def ll2xy(self, lon, lat):
        y, x = bilin_inv(lon, lat, self.lon_rho, self.lat_rho)
        return x, y





"""

Main SGrid class in roppy

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

# from netCDF4 import Dataset

from roppy.depth import sdepth, zslice, s_stretch
from roppy.sample import sample2D, bilin_inv

# ------------------------------------------------------
# Classes
# ------------------------------------------------------


class _Lazy:
    """Make lazy properties doing work only when and if needed"""

    # Recipe by Scott David Daniels
    # http://code.activestate.com/recipes/363602-lazy-property-evaluation/

    def __init__(self, calculate_function):
        self._calculate = calculate_function

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        value = self._calculate(obj)
        setattr(obj, self._calculate.__name__, value)
        return value


# ------------------------------


class SGrid:

    """Simple ROMS grid object

    Simple, minimal ROMS 3D grid object, for keeping important
    information together. Meant to be compatible with
    roms2soda's grdClass.

    Typical usage::

    >>> fid = Dataset(roms_file)
    >>> grd = SGrid(fid)

    More arguments::

    >>> fid = Dataset(roms_file)
    >>> Vinfo = {'N' : 32, 'hc' : 10, 'theta_s' : 0.8, 'theta_b' : 0.4}
    >>> grd = SGrid(fid, subgrid=(100, 121, 60, 161), Vinfo=Vinfo)

    """

    def __init__(self, ncid, subgrid=None, Vinfo=None):

        self.ncid = ncid
        self.subgrid = subgrid
        self._Vinfo = Vinfo

        self._init_horizontal()
        self._init_vertical()

    def _init_horizontal(self):

        # (sub-)grid limits
        # i0 <= i < i1, j0 <= j < j1
        Mp, Lp = self.ncid.variables["h"].shape
        if self.subgrid:
            i0 = self.subgrid[0]
            i1 = self.subgrid[1]
            j0 = self.subgrid[2]
            j1 = self.subgrid[3]
            # Allow None if no limitation
            if i0 is None:
                i0 = 0
            if i1 is None:
                i1 = Mp
            if j0 is None:
                j0 = 0
            if j1 is None:
                j1 = Lp
            # Allow negative values, relative to right/upper end
            if i0 < 0:
                i0 += Lp
            if i1 <= 0:
                i1 += Lp
            if j0 < 0:
                j0 += Mp
            if j1 <= 0:
                j1 += Mp
            self.i0, self.i1 = i0, i1
            self.j0, self.j1 = j0, j1
        else:
            self.i0, self.i1 = 0, Lp
            self.j0, self.j1 = 0, Mp

        # Shape of the grid
        self.shape = (self.j1 - self.j0, self.i1 - self.i0)

        # Slices for rho-points
        self.I = slice(self.i0, self.i1)
        self.J = slice(self.j0, self.j1)

        # U and V-points
        i0_u = max(0, self.i0 - 1)
        i1_u = min(self.i1, Lp - 1)
        j0_v = max(0, self.j0 - 1)
        j1_v = min(self.j1, Mp - 1)
        self.i0_u = i0_u
        self.j0_v = j0_v
        self.Iu = slice(i0_u, i1_u)
        self.Ju = self.J
        self.Iv = self.I
        self.Jv = slice(j0_v, j1_v)

        # Grid cell centers
        self.X = np.arange(self.i0, self.i1)
        self.Y = np.arange(self.j0, self.j1)
        # Grid cell boundaries = psi-points
        self.Xb = np.arange(self.i0 - 0.5, self.i1)
        self.Yb = np.arange(self.j0 - 0.5, self.j1)

    # ------------------------------------

    def _init_vertical(self):
        """Init vertical structure"""

        # Vinfo overrides ncid

        self.vertical = True

        if self._Vinfo:  # Explicitly given vertical info
            Vinfo = self._Vinfo
            self.N = Vinfo["N"]
            self.hc = Vinfo["hc"]
            self.Vstretching = Vinfo.get("Vstretching", 1)
            self.Vtransform = Vinfo.get("Vtransform", 1)
            self.Cs_r = s_stretch(
                self.N,
                Vinfo["theta_s"],
                Vinfo["theta_b"],
                stagger="rho",
                Vstretching=self.Vstretching,
            )
            self.Cs_w = s_stretch(
                self.N,
                Vinfo["theta_s"],
                Vinfo["theta_b"],
                stagger="w",
                Vstretching=self.Vstretching,
            )

        else:  # Vertical info from the ROMS file

            f0 = self.ncid

            try:
                self.hc = f0.variables["hc"].getValue()
                self.Cs_r = f0.variables["Cs_r"][:]
                self.Cs_w = f0.variables["Cs_w"][:]
            except KeyError:
                # No vertical information, skip the rest
                self.vertical = False

            if self.vertical:
                self.Vstretching = f0.variables.get("Vstretching", None)
                # Get S
                try:
                    self.s_rho = f0.variables["s_rho"][:]
                    self.s_w = f0.variables["s_w"][:]
                except KeyError: 
                    # No S information on file
                    self.s_rho = None
                    self.s_w = None
    
                # Vertical grid size
                self.N = len(self.Cs_r)

                # Vertical transform
                if "Vtransform" in f0.variables:
                    self.Vtransform = f0.variables["Vtransform"].getValue()
                else:
                    self.Vtransform = 1
                

    # --------------
    # Lazy reading
    # ---------------

    # Some 2D fields from the file
    # Only read if (and when) needed

    # Doing the following for a list of fields
    # @_Lazy
    # def field(self):
    #     return self.ncid.variables['field'][self.J, self.I]

    for _field in ["h", "mask_rho", "lon_rho", "lat_rho", "pm", "pn", "angle", "f"]:
        exec(
            "%s = lambda self: self.ncid.variables['%s'][self.J, self.I]"
            % (_field, _field)
        )
        exec("%s = _Lazy(%s)" % (_field, _field))

    # 3D depth structure
    @_Lazy
    def z_r(self):
        if self.vertical:
            return sdepth(
                self.h, self.hc, self.Cs_r, self.s_rho, stagger="rho", Vtransform=self.Vtransform, Vstretching=self.Vstretching
            )

    @_Lazy
    def z_w(self):
        if self.vertical:
            return sdepth(
                self.h, self.hc, self.Cs_w, self.s_w, stagger="w", Vtransform=self.Vtransform, Vstretching=self.Vstretching
            )

    # ---------------------------------
    # Wrappers for romsutil functions
    # ---------------------------------

    def zslice(self, F, z):
        if self.vertical:
            return zslice(F, self.z_r, -abs(z))

    def xy2ll(self, x, y):
        return (
            sample2D(self.lon_rho, x - self.i0, y - self.j0),
            sample2D(self.lat_rho, x - self.i0, y - self.j0),
        )

    def ll2xy(self, lon, lat):
        y, x = bilin_inv(lon, lat, self.lon_rho, self.lat_rho)
        return x + self.i0, y + self.j0

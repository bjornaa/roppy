"""ROMS Post-processing tools in python (roppy)"""

# ----------------------------------
# Init file for the roppy package
#
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# ----------------------------------

from roppy import section2
from roppy.depth import multi_zslice, s_stretch, sdepth, z_average, zslice
from roppy.fluxsec import FluxSection, staircase_from_line
from roppy.levels import nice, nice_levels
from roppy.mpl_util import landmask, levelmap
from roppy.sample import sample2D, sample2DU, sample2DV
from roppy.section import Section, linear_section
from roppy.sgrid import SGrid

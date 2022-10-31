"""ROMS Post-processing tools in python (roppy)"""

# ----------------------------------
# Init file for the roppy package
#
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# ----------------------------------

from roppy.depth import sdepth, zslice, z_average, multi_zslice, s_stretch
from roppy.sample import sample2D, sample2DU, sample2DV
from roppy.levels import nice, nice_levels
from roppy.sgrid import SGrid

from roppy.section import Section, linear_section
from roppy.fluxsec import FluxSection, staircase_from_line

from roppy import section2

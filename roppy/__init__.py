# -*- coding: utf-8 -*-

"""ROMS Post-processing tools in python (roppy)"""

# ----------------------------------
# Init file for the roppy package
#
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# ----------------------------------

from __future__ import (print_function, division,
                        absolute_import, unicode_literals)

from roppy.depth import *
from roppy.sample import *
from roppy.levels import nice, nice_levels
from roppy.sgrid import SGrid

from roppy.section import *
from roppy.fluxsec import *

from roppy import section2



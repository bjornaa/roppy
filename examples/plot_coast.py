#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Control plot of coast line generated by makecoast.py"""

# ---------
# Imports
# ---------

import numpy as np
import matplotlib.pyplot as plt

# ---------------
# User settings
# ---------------

coastfile = "data/coast.dat"

# -----------
# Read data
# -----------

X, Y = np.loadtxt(coastfile, unpack=True)

# ---------------
# Plot the data
# ---------------

plt.fill(X, Y)
plt.show()

# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys
sys.path = [".."] + sys.path
from roppy.functions import *

# ------------------------------------

class test_energy(unittest.TestCase):

    def test_constant(self):
        """The energy of a constant field is the correct constant"""
        
        imax, jmax = 4, 3
        U0 = 1.0
        V0 = 1.0
        U = np.zeros((jmax, imax+1), dtype='f') + U0
        V = np.zeros((jmax+1, imax), dtype='f') + V0
        KE = kinetic_energy(U, V)
        self.assertTrue(KE.shape == (3,4))
        self.assertEqual(KE[2,2], 0.5*(U0**2 + V0**2))

    def rest_linear(self):

        pass



# --------------------------------------

if __name__ == '__main__':
    unittest.main()

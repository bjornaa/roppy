# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys
sys.path = [".."] + sys.path
from roppy.functions import *

# ------------------------------------

class test_kinetic_energy(unittest.TestCase):

    def test_correct_shape(self):
        imax, jmax = 8, 5
        U = np.zeros((jmax, imax+1), dtype='f')
        V = np.zeros((jmax+1, imax), dtype='f')
        KE = kinetic_energy(U, V)
        self.assertTrue(KE.shape == (jmax, imax))

    def test_constant_input(self):
        """The energy of a constant field is the correct constant"""
        
        imax, jmax = 4, 3
        U0 = 1.0
        V0 = 1.0
        U = np.zeros((jmax, imax+1), dtype='f') + U0
        V = np.zeros((jmax+1, imax), dtype='f') + V0
        KE = kinetic_energy(U, V)
        self.assertEqual(KE[2,2], 0.5*(U0**2 + V0**2))

    def test_linear_input(self):
        """Correct value for linear velocity field"""

        # U(x,y) = x, V(x,y) = 0, -0.5 <= x, y <= 0.5
        # Energy = 1/24
        U = np.array([-0.5, 0.5]).reshape((1,2))
        V = np.array([ 0.0, 0.0]).reshape((2,1))
        KE = kinetic_energy(U, V)
        self.assertTrue(KE[0,0] == 1.0/24.0)

        # Exchange x and y
        # U(x,y) = 0, V(x,y) = y, -0.5 <= x, y <= 0.5
        # Energy = 1/24
        U = np.array([ 0.0, 0.0]).reshape((1,2))
        V = np.array([-0.5, 0.5]).reshape((2,1))
        KE = kinetic_energy(U, V)
        self.assertTrue(KE[0,0] == 1.0/24.0)

        # Rotate the grid and current field 45 degrees
        # U(x,y) = V(x,y) = (x+y)/sqrt(2)
        # -0.5*sqrt(2) <= x+y, x-y <= 0.5*sqrt(2)
        # Shall still have energy = 1/24
        a = 0.5/np.sqrt(2)
        def vel(x, y):
            return 2*a*(x+y)
        U = np.array([vel(-a, -a), vel(a, a)]).reshape((1,2))
        V = np.array([vel(a, -a), vel(-a, a)]).reshape((2,1))
        KE = kinetic_energy(U,V)
        self.assertAlmostEqual(KE[0,0], 1.0/24.0, places=15)
        
class test_divergence(unittest.TestCase):

    def test_correct_shape(self):
        imax, jmax = 8, 5
        U = np.zeros((jmax, imax+1), dtype='f')
        V = np.zeros((jmax+1, imax), dtype='f')
        pm = np.ones((jmax, imax))
        D = divergence(U, V, pm, pm)
        self.assertTrue(D.shape == (jmax, imax))

    def test_constant_input(self):
        """The divergence of a constant field is zero"""
        
        imax, jmax = 4, 3
        U0 = 1.0
        V0 = 0.2
        U = np.zeros((jmax, imax+1), dtype='f') + U0
        V = np.zeros((jmax+1, imax), dtype='f') + V0
        pm = np.ones((jmax, imax))
        D = divergence(U, V, pm, pm)
        self.assertEqual(D[2,2], 0.0)

    def test_rotation(self):
        """The divergence is zero in solid body rotation"""

        # U(x,y) = -y, V(x, y) = x
        #     x0 <= x <= x0+1
        #     y0 <= y <= y0+1
        def Uvel(x, y):
            return -y,
        def Vvel(x, y):
            return x
        x0 = 4.0
        y0 = 2.0
        U = np.array([Uvel(x0, y0+0.5), Uvel(x0+1, y0+0.5)]).reshape((1,2))
        V = np.array([Vvel(x0+0.5, y0), Vvel(x0+0.5, y0+1)]).reshape((2,1))
        pm = np.array([[1.0]])
        D = divergence(U, V, pm, 2*pm)
        self.assertEqual(D[0,0], 0.0)

    def test_corner(self):
        """Flow around a corner"""
        U = np.array([ 1.0, 0.0]).reshape((1,2))
        V = np.array([-1.0, 0.0]).reshape((2,1))
        pm = np.array([[1.0]])
        D = divergence(U, V, pm, pm)
        self.assertEqual(D[0,0], 0.0)

class test_div2(unittest.TestCase):
        

    def test_correct_shape(self):
        imax, jmax = 8, 5
        U = np.zeros((jmax, imax+1), dtype='f')
        V = np.zeros((jmax+1, imax), dtype='f')
        pm = np.ones((jmax+2, imax+2))
        D = div2(U, V, pm, pm)
        self.assertTrue(D.shape == (jmax, imax))

    def test_constant_input(self):
        """The divergence of a constant field is zero"""
        
        imax, jmax = 4, 3
        U0 = 1.0
        V0 = 0.2
        U = np.zeros((jmax, imax+1), dtype='f') + U0
        V = np.zeros((jmax+1, imax), dtype='f') + V0
        pm = np.ones((jmax+2, imax+2))
        pn = 2*pm
        D = div2(U, V, pm, pm)
        self.assertEqual(D[2,2], 0.0)

    def test_rotation(self):
        """The divergence is zero in solid body rotation"""

        # U(x,y) = -y, V(x, y) = x
        #     x0 <= x <= x0+1
        #     y0 <= y <= y0+1
        def Uvel(x, y):
            return -y,
        def Vvel(x, y):
            return x
        x0 = 4.0
        y0 = 2.0
        U = np.array([Uvel(x0, y0+0.5), Uvel(x0+1, y0+0.5)]).reshape((1,2))
        V = np.array([Vvel(x0+0.5, y0), Vvel(x0+0.5, y0+1)]).reshape((2,1))
        pm = np.array(9*[1.0]).reshape((3,3))
        pn = 2*pm
        D = div2(U, V, pm, pn)
        self.assertEqual(D[0,0], 0.0)

    def test_corner(self):
        """Flow around a corner"""
        U = np.array([ 1.0, 0.0]).reshape((1,2))
        V = np.array([-1.0, 0.0]).reshape((2,1))
        pm = np.array(9*[1.0]).reshape((3,3))
        pn = pm
        D = div2(U, V, pm, pn)
        self.assertEqual(D[0,0], 0.0)
        


# --------------------------------------

if __name__ == '__main__':
    unittest.main()

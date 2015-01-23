# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys
sys.path = [".."] + sys.path
#print sys.path
from roppy.functions import *

# =========================================================

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

    # Ikke isotropt
    def rest_isotropy(self):
        def vel(x, y, theta):
            a = (x*np.cos(theta) + y*np.sin(theta))
            return (a*np.cos(theta), a*np.sin(theta))
        theta = 0.1
        U = np.array([[vel(-0.5, 0.0, theta)[0],
                       vel( 0.5, 0.0, theta)[0]]])
        V = np.array([[vel(0.0, -0.5, theta)[1],
                       vel(0.0,  0.5, theta)[1]]]).T
        KE = kinetic_energy(U, V)
        print "KE = ", KE
        theta = 0.2
        U = np.array([[vel(-0.5, 0.0, theta)[0],
                       vel( 0.5, 0.0, theta)[0]]])
        V = np.array([[vel(0.0, -0.5, theta)[1],
                       vel(0.0,  0.5, theta)[1]]]).T
        KE = kinetic_energy(U, V)
        #print "KE = ", KE

        
                 
        


# ==========================================================        

class test_divergence(unittest.TestCase):
        

    def test_correct_shape(self):
        imax, jmax = 8, 5
        U = np.zeros((jmax, imax+1), dtype='f')
        V = np.zeros((jmax+1, imax), dtype='f')
        pm = np.ones((jmax+2, imax+2))
        D = divergence(U, V, pm, pm)
        self.assertTrue(D.shape == (jmax, imax))

    def test_constant_input(self):
        """The divergence of constant field in Cartesian grid is zero"""
        
        imax, jmax = 4, 3
        U0 = 1.0
        V0 = 0.2
        U = np.zeros((jmax, imax+1), dtype='f') + U0
        V = np.zeros((jmax+1, imax), dtype='f') + V0
        pm = np.ones((jmax+2, imax+2))
        pn = 2*pm
        D = divergence(U, V, pm, pm)
        self.assertEqual(D[2,2], 0.0)

    def test_rotation_1(self):
        """Solid body rotation in cartesian grid"""
        # pn = 1/dx, pm = 1/dy
        # U(x,y) = -y*dy, V(x, y) = x*dx
        # div = 0

        x, y = 4.0, 2.0
        dx, dy = 0.5, 0.5
        U = np.array([-(y-0.5)*dy, -(y+0.5)*dy]).reshape((1,2))
        V = np.array([ (x-0.5)*dx,  (x+0.5)*dx]).reshape((2,1))
        pm = (1/dx) + np.zeros((3,3))
        pn = (1/dy) + np.zeros((3,3))
        D = divergence(U, V, pm, pn)
        self.assertEqual(D[0,0], 0.0)
        
    def test_rotation_2(self):
        """Solid body rotation in polar grid"""
        # Polar coordinates (x*dx = radius, y*dy = angle)
        #   pm = 1/dx, pn = 1/(x*dx*dy)
        #   U = 0, V = x * dx
        #   div = 0

        x, y = 4.0, 2.0
        dx, dy = 0.3, 0.03
        U = np.zeros((1,2), dtype=np.float64)
        V = np.array([x*dx, x*dx]).reshape((2,1))
        pm = (1.0/dx) + np.zeros((3,3), dtype=np.float64)
        pn = (1/(dx*dy)) * np.array([1.0/(x-1), 1.0/x, 1.0/(x+1)])
        pn = np.add.outer([0,0,0], pn)
        D = divergence(U, V, pm, pn)
        self.assertEqual(D[0,0], 0.0)

    def test_radial_1(self):
        """Radial flow in Cartesian grid"""
        # U = x*dx, V = y*dy
        # pn = 1/dx, pm = 1/dy
        # div = 2
        x, y = 4, 2
        dx = 0.4
        dy = 0.5
        U = np.array([(x-0.5)*dx, (x+0.5)*dx]).reshape((1,2))
        V = np.array([(y-0.5)*dy, (y+0.5)*dy]).reshape((2,1))
        pm = (1/dx) + np.zeros((3,3))
        pn = (1/dy) + np.zeros((3,3))
        D = divergence(U, V, pm, pn)
        #print "D = ", D, dx+dy
        self.assertAlmostEqual(D[0,0], 2.0, places=15)
        
    def test_radial_2(self):
        """Radial flow"""
        # Polar coordinates (x*dx = radius, y*dy = angle)
        #   pm = 1/dx, pn = 1/(x*dx*dy)
        #   U = x*dx, V = 0
        #   div = pm*pn*(d/dx(U/pn) + d/dy(V/pm)) = 2
        x, y = 4.0, 2.0
        dx, dy = 0.3, 0.03
        U = np.array([(x-0.5)*dx, (x+0.5)*dx]).reshape((1,2))
        V = np.zeros((2,1), dtype=np.float64)
        pm = (1.0/dx) + np.zeros((3,3), dtype=np.float64)
        pn = (1/(dx*dy)) * np.array([1.0/(x-1), 1.0/x, 1.0/(x+1)])
        pn = np.add.outer([0,0,0], pn)
        D = divergence(U, V, pm, pn)
        self.assertAlmostEqual(D[0,0], 2.0, places=12)
       
    def test_vortex(self):
        """Flow around a psi-point, cartesian"""
        U = np.array([[ 0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        V = -U.T
        pm = np.ones((4,4))
        pn = pm
        D = divergence(U, V, pm, pn)
        self.assertEqual(D[0,0], 0.0)
        self.assertEqual(D[0,1], 0.0)
        self.assertEqual(D[1,0], 0.0)
        self.assertEqual(D[1,1], 0.0)

# --------------------------------------       

class test_curl(unittest.TestCase):
        
    def test_correct_shape(self):
        imax, jmax = 8, 5
        U = np.zeros((jmax, imax-1), dtype='f')
        V = np.zeros((jmax-1, imax), dtype='f')
        pm = np.ones((jmax, imax))
        C = curl(U, V, pm, pm)
        self.assertTrue(C.shape == (jmax-1, imax-1))

    def test_constant_input(self):
        """The curl of constant field in Cartesian grid is zero"""
        
        imax, jmax = 4, 3
        U0 = 1.0
        V0 = 0.2
        U = np.zeros((jmax, imax-1), dtype='f') + U0
        V = np.zeros((jmax-1, imax), dtype='f') + V0
        pm = np.ones((jmax, imax))
        pn = 2*pm
        C = curl(U, V, pm, pm)
        self.assertEqual(C[1,1], 0.0)

    def test_rotation_1(self):
        """Solid body rotation in cartesian grid"""
        # pn = 1/dx, pm = 1/dy
        # U(x,y) = -y*dy, V(x, y) = x*dx
        # curl = -2

        imax, jmax = 2, 2
        x, y = 4.0, 2.0    # Grid coordinates in lower left grid cell
        dx, dy = 0.2, 0.3  # Grid spacing
        U = np.array([-y*dy, -(y+1)*dy]).reshape((2,1))
        V = np.array([ x*dx,  (x+1)*dx]).reshape((1,2))
        pm = (1/dx) + np.zeros((jmax, imax))
        pn = (1/dy) + np.zeros((jmax, imax))
        C = curl(U, V, pm, pn)
        self.assertEqual(C[0,0], -2.0)
        
    def test_rotation_2(self):
        """Solid body rotation in polar grid"""
        # Polar coordinates (x*dx = radius, y*dy = angle)
        #   pm = 1/dx, pn = 1/(x*dx*dy)
        #   U = 0, V = x * dx
        #   curl = -2
        # Errors when curving a lot

        imax, jmax = 2, 2
        x, y = 1014.0, 620.0
        dx, dy = 0.001, 0.001     # 1 km resolution
        U = np.zeros((2,1), dtype=np.float64)
        V = np.array([x*dx, (x+1)*dx]).reshape((1,2))
        pm = (1.0/dx) + np.zeros((2,2), dtype=np.float64)
        pn = (1/(dx*dy)) * np.array([[1.0/x, 1.0/(x+1)], [1.0/x, 1.0/(x+1)]])
        C = curl(U, V, pm, pn)
        self.assertAlmostEqual(C[0,0], -2.0, places=6)

    def test_radial_1(self):
        """Radial flow in Cartesian grid"""
        # U = x*dx, V = y*dy
        # pn = 1/dx, pm = 1/dy
        # curl = 0
        imax, jmax = 2, 2
        x, y = 4, 2
        dx = 0.4
        dy = 0.5
        U = np.array([x*dx, x*dx]).reshape((2,1))
        V = np.array([y*dy, y*dy]).reshape((1,2))
        pm = (1/dx) + np.zeros((jmax, imax))
        pn = (1/dy) + np.zeros((jmax, imax))
        C = curl(U, V, pm, pn)
        self.assertEqual(C[0,0], 0.0)
        
    def test_radial_2(self):
        """Radial flow"""
        # Polar coordinates (x*dx = radius, y*dy = angle)
        #   pm = 1/dx, pn = 1/(x*dx*dy)
        #   U = x*dx, V = 0
        #   curl = 0+
        imax, jmax = 2, 2
        x, y = 4.0, 2.0
        dx, dy = 0.3, 0.03
        U = np.array([x*dx, x*dx]).reshape((2,1))
        V = np.zeros((1,2), dtype=np.float64)
        pm = (1.0/dx) + np.zeros((jmax, imax), dtype=np.float64)
        pn = (1/(dx*dy)) * np.array([[1.0/x, 1.0/(x+1)], [1.0/x, 1.0/(x+1)]])
        C = curl(U, V, pm, pn)
        #print C.shape
        self.assertEqual(C[0,0], 0.0)
       
    def test_vortex(self):
        """Flow around a psi-point, cartesian"""
        imax, jmax = 2, 2
        U = np.array([1.0, -1.0]).reshape((2,1))
        V = np.array([-1.0, 1.0]).reshape((1,2))
        pm = np.ones((jmax, imax))
        pn = pm
        C = curl(U, V, pm, pn)
        self.assertEqual(C[0,0], -4.0)


# --------------------------------------

if __name__ == '__main__':
    unittest.main()

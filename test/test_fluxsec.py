import unittest
import numpy as np

import sys
#sys.path.append('..')
sys.path = ['..'] + sys.path
#sys.path = ['..'] 

from roppy.fluxsec import *

class FakeGrid(object):

    def __init__(self, imax=20, jmax=16, kmax=10):

        # Depth = constant = 100 m
        self.h = 100.0 + np.zeros((jmax, imax))
        self.hc = 10.0
        self.Cs_w = np.linspace(-1, 0, kmax+1)
        self.Cs_r = self.Cs_w[1:] - self.Cs_w[:-1]
        self.mask_rho = np.ones((jmax, imax))     # No land
        self.Vtransform = 1                         
        # dx = dy = 1000 m
        self.pm = 0.001 + np.zeros((jmax, imax))
        self.pn = 0.001 + np.zeros((jmax, imax))

# ------------------------------------

class TestFluxSec(unittest.TestCase):
    """Testing setup of FluxSection object"""

    def test_attributes(self):
        """Attributes gets the correct values"""

        imax, jmax, kmax = 20, 16, 10
        i0, j0 = 4, 12
        i1, j1 = 7, 14

        I = np.array([4, 5, 5, 6, 6, 7])
        J = np.array([12, 12, 13, 13, 14, 14])
        grd = FakeGrid(imax, jmax, kmax)
        sec = FluxSection(grd, I, J)

        # Dimensions
        self.assertEqual(sec.N, kmax)
        self.assertTrue(len(sec) == len(I)-1)

        # Positions of midpoints on edges
        Xpos = [4, 4.5, 5, 5.5, 6]
        Ypos = [11.5, 12, 12.5, 13, 13.5]
        self.assertTrue(np.all(sec.X == Xpos))
        self.assertTrue(np.all(sec.Y == Ypos))
        
        # edges 1 and 3 are U (counting from zero)
        U_edge = [False, True, False, True, False]
        V_edge = np.logical_not(U_edge)
        self.assertTrue(np.all(sec.Eu == U_edge))
        self.assertTrue(np.all(sec.Ev == V_edge))
        # Positive flux directions are down and to the right
        directions = np.array([-1, 1, -1, 1, -1])
        self.assertTrue(np.all(sec.dir == directions))

        # grid geometry
        self.assertTrue(np.all(sec.dS == 1000))
        self.assertTrue(np.all(sec.h == 100.0))

        # Vertically integrated
        self.assertTrue(np.all(np.sum(sec.dZ, axis=0) == sec.h))
        self.assertTrue(np.all(np.sum(sec.dSdZ, axis=0) == 1e5))
        
    def test_closed_circuit(self):
        """Divergence free current trough a cloced 'section' gives zero flux"""

        grd = FakeGrid(3, 3, 5)
        a, b = 1.0, 2.0
        U = np.zeros((5, 3, 2))
        U[:, :, 0] = a
        U[:, :, 1] = b
        V = np.zeros((5, 2, 3))
        V[:, 0, :] = b
        V[:, 1, :] = a
        # Divergence of grid cell (1,1) is zero
        div = -V[-1, 0, 1] + U[-1, 1, 1] + V[-1, 1, 1] - U[-1, 1, 0]
        self.assertEqual(div, 0.0)
        
        I = np.array([1, 2, 2, 1, 1])
        J = np.array([1, 1, 2, 2, 1])

        sec = FluxSection(grd, I, J)
        Fnet, Fout = sec.transport(U, V)
        self.assertEqual(Fnet, 0.0)
        self.assertEqual(Fout, (a+b)*1e5)

        
        
    def test_flux_alignment(self):
        """Using correct velocities to compute the flux"""

        imax, jmax, kmax = 20, 16, 10
        i0, j0 = 4, 12
        i1, j1 = 7, 14

        I = np.array([4, 5, 5, 6, 6, 7])
        J = np.array([12, 12, 13, 13, 14, 14])
        grd = FakeGrid(imax, jmax, kmax)
        sec = FluxSection(grd, I, J)
        sec2 = FluxSection(grd, I[::-1], J[::-1])  # Reversed section

        U = np.zeros((kmax, jmax, imax-1))
        V = np.zeros((kmax, jmax-1, imax))

        a, b = 1, 2
        # grid cell (4, 12) -> (5, 12): u-point (5, 12)
        U[:, 12, 4] = a
        # grid cell (5, 13) -> (6, 13): u-point (6, 13)
        U[:, 13, 5] = b

        flux,flux1 = sec.transport(U,V)
        self.assertEqual(flux, (a+b)*1e5)
        flux,flux1 = sec2.transport(U,V)
        self.assertEqual(flux, -(a+b)*1e5)

        U = np.zeros((kmax, jmax, imax-1))
        a, b, c = 1, 2, 4
        # grid cell (4, 11) -> (4, 12): v-point (4, 12)
        V[:, 11, 4] = a
        # grid cell (5, 12) -> (5, 13): v-point (5, 13)
        V[:, 12, 5] = b
        # grid cell (6, 13) -> (6, 14): v-point (6, 14)
        V[:, 13, 6] = c
        flux,flux1 = sec.transport(U, V)
        self.assertEqual(flux, -(a+b+c)*1e5)
        flux,flux1 = sec2.transport(U, V)
        self.assertEqual(flux, (a+b+c)*1e5)
        


# ------------------------------------

class TestStaircase(unittest.TestCase):
    """Testing the function staircase from line"""


    def test_nosec(self):
        """Section is a single point"""

        i0, j0 = 15, 7
        i1, j1 = 15, 7

        self.assertRaises(ValueError, staircase_from_line, i0, i1, j0, j1)
                               

    def test_X(self):
        """A section in the X direction"""

        i0, j0 = 14, 4
        i1, j1 = 17, 4

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == np.array([14, 15, 16, 17])))
        self.assertTrue(np.all(Y == np.array([4, 4, 4, 4])))
                               
    def test_Y(self):
        """A section in the Y direction"""

        i0, j0 = 14, 4
        i1, j1 = 14, 7

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == np.array([14, 14, 14, 14])))
        self.assertTrue(np.all(Y == np.array([4, 5, 6, 7])))

    def test_XY(self):
        """A diagonal section"""

        i0, j0 = 14, 4
        i1, j1 = 17, 7

        X, Y = staircase_from_line(i0, i1, j0, j1)
        self.assertTrue(np.all(X ==
                               np.array([14, 15, 15, 16, 16, 17, 17])))
        self.assertTrue(np.all(Y ==
                               np.array([4, 4, 5, 5, 6, 6, 7])))
                               
    def test_Xdir_Xinc_Yinc(self):
        """X-direction, X, Y increasing"""

        i0, j0 = 14, 4
        i1, j1 = 17, 6

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X ==
                               np.array([14, 15, 15, 16, 16, 17])))
        self.assertTrue(np.all(Y ==
                               np.array([4, 4, 5, 5, 6, 6])))
                               
    def test_Xdir_Xinc_Ydec(self):

        i0, j0 = 14, 6
        i1, j1 = 17, 4

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X ==
                               np.array([14, 15, 15, 16, 16, 17])))
        self.assertTrue(np.all(Y ==
                               np.array([6, 6, 5, 5, 4, 4])))
                               
    def test_Xdir_Xdec_Yinc(self):

        i0, j0 = 17, 4
        i1, j1 = 14, 6

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X ==
                               np.array([17, 16, 16, 15, 15, 14])))
        self.assertTrue(np.all(Y ==
                               np.array([4, 4, 5, 5, 6, 6])))
        
    def test_Xdir_Xdec_Ydec(self):

        i0, j0 = 17, 6
        i1, j1 = 14, 4

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X ==
                               np.array([17, 16, 16, 15, 15, 14])))
        self.assertTrue(np.all(Y ==
                               np.array([6, 6, 5, 5, 4, 4])))

    def test_Ydir_Xinc_Yinc(self):

        i0, j0 = 14, 4
        i1, j1 = 16, 7

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X ==
                               np.array([14, 14, 15, 15, 16, 16])))
        self.assertTrue(np.all(Y ==
                               np.array([4, 5, 5, 6, 6, 7])))

    def test_Ydir_Xinc_Ydec(self):

        i0, j0 = 14, 7
        i1, j1 = 16, 4

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X ==
                               np.array([14, 14, 15, 15, 16, 16])))
        self.assertTrue(np.all(Y ==
                               np.array([7, 6, 6, 5, 5, 4])))
                               
    def test_Ydir_Xdec_Yinc(self):

        i0, j0 = 16, 4
        i1, j1 = 14, 7

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X ==
                               np.array([16, 16, 15, 15, 14, 14])))
        self.assertTrue(np.all(Y ==
                               np.array([4, 5, 5, 6, 6, 7])))
                               
    def test_Ydir_Xdec_Ydec(self):

        i0, j0 = 16, 7
        i1, j1 = 14, 4

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X ==
                               np.array([16, 16, 15, 15, 14, 14])))
        self.assertTrue(np.all(Y ==
                               np.array([7, 6, 6, 5, 5, 4])))


        


# --------------------------------------

if __name__ == '__main__':
    unittest.main()

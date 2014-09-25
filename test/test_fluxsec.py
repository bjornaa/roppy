import unittest
import numpy as np

import sys
#sys.path.append('..')
sys.path = ['..'] + sys.path
#sys.path = ['..'] 

from roppy.fluxsec import staircase_from_line

# A synthetic grid object for testing
class MyGrid(object):

    def __init__(self):

        imax = 20
        jmax = 16
        # Depth = constant = 100 m
        self.h = 100.0 + np.zeros((jmax, imax))
        self.hc = 10.0
        self.Cs_w = np.linspace(-1, 0, 10)
        self.Cs_r = self.Cs_w[1:] - self.Cs_w[:-1]
        self.mask_rho = np.ones((jmax, imax))
        self.Vtransform = 1
        # dx = dy = 1000 m
        self.pm = 0.001 + np.zeros((jmax, imax))
        self.pn = 0.001 + np.zeros((jmax, imax))

# ------------------------------------

class TestStaircase(unittest.TestCase):


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

        grd = MyGrid()
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

    def test_kake(self):
        i0, j0 = 1, 11
        i1, j1 = 8, 12
        X, Y = staircase_from_line(i0, i1, j0, j1)
        print X,  Y
        

        


# --------------------------------------

if __name__ == '__main__':
    unittest.main()

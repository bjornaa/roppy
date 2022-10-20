import unittest
import numpy as np

import sys
sys.path.append('..')

from roppy.section import linear_section

# A synthetic grid object for testing
class MyGrid(object):

    def __init__(self):

        imax = 20
        jmax = 16
        # Depth = constant = 100 m
        self.h = 100.0 + np.zeros((jmax, imax))
        self.hc = 10.0
        self.N = 9
        self.Cs_w = np.linspace(-1, 0, 10)
        self.Cs_r = self.Cs_w[1:] - self.Cs_w[:-1]
        self.s_w = np.linspace(-1, 0, 10)
        self.s_rho = self.Cs_r
        self.mask_rho = np.ones((jmax, imax))
        self.Vtransform = 1
        # dx = dy = 1000 m
        self.pm = 0.001 + np.zeros((jmax, imax))
        self.pn = 0.001 + np.zeros((jmax, imax))

# ------------------------------------

class TestLinearSection(unittest.TestCase):

    def test_X(self):
        """A section in the X direction"""

        grd = MyGrid()
        i0, j0 = 14, 4
        i1, j1 = 17, 4

        sec = linear_section(i0, i1, j0, j1, grd)

        self.assertTrue(np.all(sec.X == np.array([14, 15, 16, 17])))
        self.assertTrue(np.all(sec.Y == np.array([4, 4, 4, 4])))
                               
    def test_Y(self):
        """A section in the Y direction"""

        grd = MyGrid()
        i0, j0 = 14, 4
        i1, j1 = 14, 7

        sec = linear_section(i0, i1, j0, j1, grd)

        self.assertTrue(np.all(sec.X == np.array([14, 14, 14, 14])))
        self.assertTrue(np.all(sec.Y == np.array([4, 5, 6, 7])))

    def test_XY(self):
        """A diagonal section"""

        grd = MyGrid()
        i0, j0 = 14, 4
        i1, j1 = 17, 7

        sec = linear_section(i0, i1, j0, j1, grd)

        self.assertTrue(np.all(sec.X == np.array([14, 15, 16, 17])))
        self.assertTrue(np.all(sec.Y == np.array([4, 5, 6, 7])))
                               
    def test_Xminus(self):
        """Skew section, negative X direction"""

        grd = MyGrid()
        i0, j0 = 17, 4
        i1, j1 = 14, 5

        sec = linear_section(i0, i1, j0, j1, grd)

        self.assertTrue(np.all(sec.X == np.array([17, 16, 15, 14])))
        self.assertTrue(np.allclose(sec.Y, np.array([4, 4.33, 4.67, 5]),
                                    atol=0.005))
                               
            
    def test_Yminus(self):
        """Skew section, negative Y direction"""

        grd = MyGrid()
        i0, j0 = 15, 7
        i1, j1 = 14, 4

        sec = linear_section(i0, i1, j0, j1, grd)

        self.assertTrue(np.allclose(sec.X, np.array([15, 14.67, 14.33, 14]),
                                    atol=0.005))
        self.assertTrue(np.all(sec.Y == np.array([7, 6, 5, 4])))
                               
            
    def test_Yminus(self):
        """Skew section, negative Y direction"""

        grd = MyGrid()
        i0, j0 = 15, 7
        i1, j1 = 14, 4

        sec = linear_section(i0, i1, j0, j1, grd)

        self.assertTrue(np.allclose(sec.X, np.array([15, 14.67, 14.33, 14]),
                                    atol=0.005))
        self.assertTrue(np.all(sec.Y == np.array([7, 6, 5, 4])))
                               
            
    def test_Yminus(self):
        """Skew section, negative Y direction"""

        grd = MyGrid()
        i0, j0 = 15, 7
        i1, j1 = 14, 4

        sec = linear_section(i0, i1, j0, j1, grd)

        self.assertTrue(np.allclose(sec.X, np.array([15, 14.67, 14.33, 14]),
                                    atol=0.005))
        self.assertTrue(np.all(sec.Y == np.array([7, 6, 5, 4])))
                               
            
    def test_nosec(self):
        """Section is a single point"""

        grd = MyGrid()
        i0, j0 = 15, 7
        i1, j1 = 15, 7

        self.assertRaises(ValueError, linear_section, i0, i1, j0, j1, grd)
                               
            





# --------------------------------------

if __name__ == '__main__':
    unittest.main()

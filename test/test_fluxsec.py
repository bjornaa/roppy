import unittest
import numpy as np

from roppy.fluxsec import *

# ------------------------------------------


class FakeGrid(object):
    """An idealized grid class for testing"""

    def __init__(self, imax=20, jmax=16, kmax=10):

        # Depth = constant = 100 m
        self.h = 100.0 + np.zeros((jmax, imax))
        self.hc = 10.0
        self.Cs_w = np.linspace(-1, 0, kmax+1)
        self.Cs_r = self.Cs_w[1:] - self.Cs_w[:-1]
        self.s_w = np.linspace(-1, 0, kmax+1)
        self.mask_rho = np.ones((jmax, imax))     # No land
        self.Vtransform = 1
        self.Vstretching = None
        # dx = dy = 1000 m
        self.pm = 0.001 + np.zeros((jmax, imax))
        self.pn = 0.001 + np.zeros((jmax, imax))
        # no offset
        self.i0 = 0
        self.j0 = 0


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

        imax, jmax, kmax = 20, 16, 5
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

        flux, flux1 = sec.transport(U, V)
        self.assertEqual(flux, (a+b)*1e5)
        flux, flux1 = sec2.transport(U, V)
        self.assertEqual(flux, -(a+b)*1e5)

        U = np.zeros((kmax, jmax, imax-1))  # reset U
        a, b, c = 1, 2, 4
        # grid cell (4, 11) -> (4, 12): v-point (4, 12)
        V[:, 11, 4] = a
        # grid cell (5, 12) -> (5, 13): v-point (5, 13)
        V[:, 12, 5] = b
        # grid cell (6, 13) -> (6, 14): v-point (6, 14)
        V[:, 13, 6] = c
        flux, flux1 = sec.transport(U, V)
        self.assertEqual(flux, -(a+b+c)*1e5)
        flux, flux1 = sec2.transport(U, V)
        self.assertEqual(flux, (a+b+c)*1e5)

# ------------------------------------


class TestAnalytical(unittest.TestCase):
    """Analytically defined testcases"""

    def test_channel(self):
        """Uniform channel flow"""

        # Make a channel
        imax, jmax, kmax = 20, 10, 5
        grd = FakeGrid(imax, jmax, kmax)

        # Constant current 1 m/s along channel
        U = np.ones((kmax, jmax, imax-1))
        V = np.zeros((kmax, jmax-1, imax-1))

        # Flux depends only on j1-j0
        i0, j0 = 2, 2
        i1, j1 = 8, 5
        I, J = staircase_from_line(i0, i1, j0, j1)
        sec = FluxSection(grd, I, J)
        Fnet, Fright = sec.transport(U, V)
        self.assertEqual(Fnet, (j1-j0)*1e5)
        self.assertEqual(Fright, (j1-j0)*1e5)

        i0, j0 = 10, 2
        i1, j1 = 11, 5
        I, J = staircase_from_line(i0, i1, j0, j1)
        sec = FluxSection(grd, I, J)
        Fnet, Fright = sec.transport(U, V)
        self.assertEqual(Fnet, (j1-j0)*1e5)
        self.assertEqual(Fright, (j1-j0)*1e5)

    def test_shear_current(self):

        # Make a channel
        imax, jmax, kmax = 20, 10, 5
        grd = FakeGrid(imax, jmax, kmax)

        # Make a shear current, in X-direction
        U = np.empty((kmax, jmax, imax-1))
        U[:, :, :] = np.arange(jmax)[None, :, None]
        V = np.zeros((kmax, jmax-1, imax-1))

        # Flux depends only on the j-values
        i0, j0 = 2, 2
        i1, j1 = 8, 5
        I, J = staircase_from_line(i0, i1, j0, j1)

        sec = FluxSection(grd, I, J)
        Fnet, Fright = sec.transport(U, V)
        anaflux = 1.0e5 * np.arange(j0, j1).sum()
        self.assertEqual(Fnet, anaflux)

    def test_solid_body_rotation(self):

        # Make a grid
        imax, jmax, kmax = 20, 10, 5
        x0, y0 = 8, 5
        grd = FakeGrid(imax, jmax, kmax)

        # Solid body rotation
        # U(x,y) = -(y-y0), V(x,y) = x-x0

        # U[j,i] lives on (x,y) = (i+0.5, j)
        U = np.empty((kmax, jmax, imax-1))
        J = np.arange(0, jmax)
        U[:, J, :] = - (J[None, :, None] - y0)

        # V[j,i] lives on (x,y) = (i, j+0.5)
        V = np.empty((kmax, jmax-1, imax))
        I = np.arange(0, imax)
        V[:, :, I] = I - x0

        # ROMS psi-indices of symmetric section
        # End points x = x0 +/- 1.5, y = y0 +/- 1.5
        I = [7, 8, 8, 9, 9, 10, 10]
        J = [4, 4, 5, 5, 6,  6,  7]

        sec = FluxSection(grd, I, J)
        F, F1 = sec.transport(U, V)
        # Zero total flux
        self.assertEqual(F, 0.0)
        # Two contibutions to positive flux
        pos_flux = (-V[-1, 3, 7] + U[-1, 4, 6]) * 1e5
        self.assertEqual(F1, pos_flux)

    def test_path_independence(self):
        """Flux is path independent in a non-divergent flow"""

        imax, jmax, kmax = 20, 15, 5

        grd = FakeGrid(imax, jmax, kmax)

        # Make an "arbitrary" non-constant stream function
        def f(y, x):
            return np.sin(0.5*x + 0.8*y)
        psi = np.fromfunction(f, (jmax-1, imax-1))

        # Compute the non-divergent (curl) flow field
        U = np.zeros((kmax, jmax, imax-1))
        V = np.zeros((kmax, jmax-1, imax))
        U[:, :-1, :] += psi[None, :, :]
        U[:, 1:, :]  -= psi[None, :, :]
        V[:, :, :-1] -= psi[None, :, :]
        V[:, :, 1:]  += psi[None, :, :]

        # Take three psi-points
        i0, j0 = 3, 2
        i1, j1 = 15, 6
        i2, j2 = 6, 4

        # Make the three sections
        sec01 = FluxSection(grd, *staircase_from_line(i0, i1, j0, j1))
        sec12 = FluxSection(grd, *staircase_from_line(i1, i2, j1, j2))
        sec02 = FluxSection(grd, *staircase_from_line(i0, i2, j0, j2))

        # Compute the fluxes
        F01, _ = sec01.transport(U, V)
        F12, _ = sec12.transport(U, V)
        F02, _ = sec02.transport(U, V)

        # Flux across path from point zero to two is independent of path
        self.assertAlmostEqual(F01+F12, F02, places=10)

# ------------------------------------


class TestSubGrid(unittest.TestCase):


    def test_shear_current(self):

        # Make a channel
        imax, jmax, kmax = 20, 10, 5
        grd = FakeGrid(imax, jmax, kmax)

        # Make a shear current, in X-direction
        U = np.empty((kmax, jmax, imax-1))
        U[:, :, :] = np.arange(jmax)[None, :, None]
        V = np.zeros((kmax, jmax-1, imax-1))

        # Flux depends only on the j-values
        i0, j0 = 6, 3
        i1, j1 = 12, 5
        I, J = staircase_from_line(i0, i1, j0, j1)

        # Fake a subgrid, and adjust velocity accordingly
        grd.i0 = 3
        grd.j0 = 1
        U = U[:, grd.j0:, grd.i0-1:]
        V = V[:, grd.j0-1:, grd.i0:]

        sec = FluxSection(grd, I, J)
        Fnet, Fright = sec.transport(U, V)
        anaflux = 1.0e5 * np.arange(j0, j1).sum()
        self.assertEqual(Fnet, anaflux)

# ---------------------------------------------------


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

        self.assertTrue(np.all(X == [14, 15, 16, 17]))
        self.assertTrue(np.all(Y == [4, 4, 4, 4]))

    def test_Y(self):
        """A section in the Y direction"""

        i0, j0 = 14, 4
        i1, j1 = 14, 7

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [14, 14, 14, 14]))
        self.assertTrue(np.all(Y == [4, 5, 6, 7]))

    def test_XY(self):
        """A diagonal section"""

        i0, j0 = 14, 4
        i1, j1 = 17, 7

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [14, 15, 15, 16, 16, 17, 17]))
        self.assertTrue(np.all(Y == [4, 4, 5, 5, 6, 6, 7]))

    def test_Xdir_Xinc_Yinc(self):
        """X-direction, X, Y increasing"""

        i0, j0 = 14, 4
        i1, j1 = 17, 6

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [14, 15, 15, 16, 16, 17]))
        self.assertTrue(np.all(Y == [4, 4, 5, 5, 6, 6]))

    def test_Xdir_Xinc_Ydec(self):

        i0, j0 = 14, 6
        i1, j1 = 17, 4

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [14, 15, 15, 16, 16, 17]))
        self.assertTrue(np.all(Y == [6, 6, 5, 5, 4, 4]))

    def test_Xdir_Xdec_Yinc(self):

        i0, j0 = 17, 4
        i1, j1 = 14, 6

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [17, 16, 16, 15, 15, 14]))
        self.assertTrue(np.all(Y == [4, 4, 5, 5, 6, 6]))

    def test_Xdir_Xdec_Ydec(self):

        i0, j0 = 17, 6
        i1, j1 = 14, 4

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [17, 16, 16, 15, 15, 14]))
        self.assertTrue(np.all(Y == [6, 6, 5, 5, 4, 4]))

    def test_Ydir_Xinc_Yinc(self):

        i0, j0 = 14, 4
        i1, j1 = 16, 7

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [14, 14, 15, 15, 16, 16]))
        self.assertTrue(np.all(Y == [4, 5, 5, 6, 6, 7]))

    def test_Ydir_Xinc_Ydec(self):

        i0, j0 = 14, 7
        i1, j1 = 16, 4

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [14, 14, 15, 15, 16, 16]))
        self.assertTrue(np.all(Y == [ 7,  6,  6,  5,  5,  4]))

    def test_Ydir_Xdec_Yinc(self):

        i0, j0 = 16, 4
        i1, j1 = 14, 7

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [16, 16, 15, 15, 14, 14]))
        self.assertTrue(np.all(Y == [4, 5, 5, 6, 6, 7]))

    def test_Ydir_Xdec_Ydec(self):

        i0, j0 = 16, 7
        i1, j1 = 14, 4

        X, Y = staircase_from_line(i0, i1, j0, j1)

        self.assertTrue(np.all(X == [16, 16, 15, 15, 14, 14]))
        self.assertTrue(np.all(Y == [7, 6, 6, 5, 5, 4]))

# ---------------------------------

class TestSampling(unittest.TestCase):
    """Testing the scalar sampling methods"""

    def test_X(self):
        """Correct if F = X"""
        imax, jmax, kmax = 10, 8, 5
        grd = FakeGrid(imax, jmax, kmax)

        # F[j,i] = i
        F = np.empty((jmax, imax))
        F[:, :] = np.arange(imax)[None,:]

        # First a u-point at (1.5, 2)
        # thereafter a v-point at (2, 2.5)
        I = [2, 2, 3]
        J = [1, 2, 2]
        sec = FluxSection(grd, I, J)
        self.assertTrue(np.all(sec.sample2D(F) == sec.X))

    def test_linear(self):
        """Correct for F linear in X and Y"""

        imax, jmax, kmax = 10, 8, 5
        grd = FakeGrid(imax, jmax, kmax)

        a, b = 2.4, 1.9
        F = np.empty((jmax, imax))
        F[:, :] = np.fromfunction(lambda j, i: a*i + b*j, (jmax, imax))

        i0, j0 = 1, 1
        i1, j1 = 5, 3
        sec = FluxSection(grd, *staircase_from_line(i0,i1,j0,j1))
        self.assertTrue(np.all(np.abs(
            sec.sample2D(F) - (a*sec.X + b*sec.Y)) < 1.e-14))

    def test_linear2(self):
        """Get all combinations of X and Y directions"""

        imax, jmax, kmax = 10, 8, 5
        grd = FakeGrid(imax, jmax, kmax)

        a, b = 2.4, 1.9
        F = np.empty((jmax, imax))
        F[:, :] = np.fromfunction(lambda j, i: a*i + b*j, (jmax, imax))

        I = [3, 4, 4, 3, 3]
        J = [1, 1, 2, 2, 1]
        sec = FluxSection(grd, I, J)
        self.assertTrue(np.all(np.abs(
            sec.sample2D(F) - (a*sec.X + b*sec.Y)) < 1.e-14))


# --------------------------------------

if __name__ == '__main__':
    unittest.main()

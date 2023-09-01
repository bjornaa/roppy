
"""Unit tests for the vertical depth functions in roppy/depth.py"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# ----------------------------------

import unittest

import numpy as np

from roppy import multi_zslice, s_stretch, z_average, zslice

# ------------------------------------

# ------------------------


class test_s_stretch(unittest.TestCase):
    def test_valid_output(self):
        """For rho-points C should obey -1 < C[i] < C[i+1] < 0"""
        N = 30
        theta_s = 6.0
        theta_b = 0.6
        C = s_stretch(N, theta_s, theta_b)
        Cp = np.concatenate(([-1], C, [0]))
        D = np.diff(Cp)
        self.assertTrue(np.all(D > 0))
        self.assertEqual(len(C), N)

    def test_valid_output_w(self):
        """For w-points C should obey -1 < C[i+1] < 0"""
        N = 30
        theta_s = 6.0
        theta_b = 0.6
        C = s_stretch(N, theta_s, theta_b, stagger="w")
        # Check increasing
        self.assertTrue(np.all(np.diff(C) > 0))
        # End points
        self.assertEqual(C[0], -1.0)
        self.assertEqual(C[-1], 0.0)
        # Length
        self.assertEqual(len(C), N + 1)


class test_Vstretching4(unittest.TestCase):
    def test_valid_output_4(self):
        """For rho-points C should obey -1 < C[i] < C[i+1] < 0"""
        N = 30
        theta_s = 6.0
        theta_b = 0.6
        C = s_stretch(N, theta_s, theta_b, Vstretching=4)
        Cp = np.concatenate(([-1], C, [0]))
        D = np.diff(Cp)
        self.assertTrue(np.all(D > 0))
        self.assertEqual(len(C), N)

    def test_valid_output(self):
        """For w-points C should obey -1 < C[i+1] < 0"""
        N = 30
        theta_s = 6.0
        theta_b = 0.6
        C = s_stretch(N, theta_s, theta_b, Vstretching=4, stagger="w")
        # Check increasing
        self.assertTrue(np.all(np.diff(C) > 0))
        # End points
        self.assertEqual(C[0], -1.0)
        self.assertEqual(C[-1], 0.0)
        # Length
        self.assertEqual(len(C), N + 1)


# ----------------------------------------------------


class Test_multi_zslice(unittest.TestCase):
    def test_scalar(self):
        """With single scalar, should equal zslice"""

        K = np.linspace(-95, -5, 10)
        z_r = [[K, K], [2 * K, 0.7 * K], [K, 1.2 * K]]
        z_r = np.transpose(z_r)  # shape = 10,2,3 with depth axis first
        F = np.random.random(z_r.shape)  # array of shape 10,2,3
        z = -52.0  # z is scalar
        self.assertTrue(np.all(multi_zslice(F, z_r, z) == zslice(F, z_r, z)))

    def test_single(self):
        """Equals zslice for a single depth level"""
        K = np.linspace(-95, -5, 10)
        z_r = [[K, K], [2 * K, 0.7 * K], [K, 1.2 * K]]
        z_r = np.transpose(z_r)  # shape = 10,2,3 with depth axis first
        F = np.random.random(z_r.shape)  # array of shape 10,2,3
        # z array, shape = 2, 3
        z = np.array([[-32, -12.3, -120.0], [-22.2, -33.3, -72]])
        self.assertTrue(np.all(multi_zslice(F, z_r, z) == zslice(F, z_r, z)))
        z1 = np.array(z)[None, :, :]
        self.assertTrue(np.all(multi_zslice(F, z_r, z1) == zslice(F, z_r, z)))

    def test_interp(self):
        """multi_zslice of 1D field is np.interp"""

        z_r = np.linspace(-95, -5, 10)
        F = np.random.random(z_r.shape)
        z = [-112, -90, -72.3, -52.2, -18, -2]
        A1 = multi_zslice(F, z_r, z)
        A2 = np.interp(z, z_r, F)
        self.assertTrue(np.allclose(A1, A2))

    def test_multiple_zslice(self):
        """Give samme result as multiple zslices"""

        K = np.linspace(-95, -5, 10)
        z_r = [[K, K], [2 * K, 0.7 * K], [K, 1.2 * K]]
        z_r = np.transpose(z_r)  # shape = 10,2,3 with depth axis first
        F = np.random.random(z_r.shape)  # array of shape 10,2,3
        Z = [-52.0, -40.0, -22.3, -1]
        Fmz = multi_zslice(F, z_r, Z)
        for k, z in enumerate(Z):
            Fz = zslice(F, z_r, z)
            self.assertTrue(np.allclose(Fz, Fmz[k]))

    def test_unsorted(self):
        """The depths can be unsorted"""

        z_r = np.linspace(-95, -5, 10)
        F = np.random.random(z_r.shape)
        z = np.array([-112, -90, -72.3, -52.2, -18, -2])
        I = [2, 0, 3, 1, 5, 4]
        A1 = multi_zslice(F, z_r, z)
        A2 = multi_zslice(F, z_r, z[I])
        self.assertTrue(np.allclose(A1[I], A2))


# Ha noen tester med array-shape
#

# -------------------------------------------------


class Test_z_average(unittest.TestCase):
    def test_constant(self):
        """Average of constant field returns the constant"""
        const = 3.14
        z_r = np.linspace(-95, -5, 10)
        F = np.zeros_like(z_r) + const
        z0, z1 = -52.0, -33.3
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), const)
        z0 = -52  # Test integer depth
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), const)

    def test_linear(self):
        """z_average is linear in the field F"""
        a, b = 3.8, 11.2
        z_r = np.linspace(-95, -5, 10)
        F1 = np.random.random(z_r.shape)
        F2 = np.random.random(z_r.shape)
        F = a * F1 + b * F2
        z0, z1 = -52.0, -22.2
        v = z_average(F, z_r, z0, z1)
        v1 = z_average(F1, z_r, z0, z1)
        v2 = z_average(F2, z_r, z0, z1)
        self.assertAlmostEqual(v, a * v1 + b * v2)

    def test_identity(self):
        """z_average of z_r is mid-point"""
        z_r = np.linspace(-95, -5, 10)
        F = z_r
        z0, z1 = -82.0, -55.0  # Floats
        A = 0.5 * (z0 + z1)
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), A)
        z0, z1 = -82, -55  # Integers
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), A)

    def test_none_between(self):
        """Correct for z_r[k-1] <= z0 < z1 <= z_r[k]"""
        z_r = np.linspace(-95, -5, 10)
        F = np.random.random(z_r.shape)
        a0, b0 = 0.3, 0.7  # sum = 1
        a1, b1 = 0.6, 0.4  # sum = 1
        k = 4
        z0 = a0 * z_r[k - 1] + b0 * z_r[k]
        z1 = a1 * z_r[k - 1] + b1 * z_r[k]
        f0 = a0 * F[k - 1] + b0 * F[k]
        f1 = a1 * F[k - 1] + b1 * F[k]
        A = 0.5 * (f0 + f1)
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), A)

    def test_one_between(self):
        """Correct for z_r[k-1] <= z0 < z_r[k] <= z1 < z_r[k]"""
        z_r = np.linspace(-95, -5, 10)
        F = np.random.random(z_r.shape)
        a0, b0 = 0.3, 0.7  # sum = 1
        a1, b1 = 0.6, 0.4  # sum = 1
        k = 4
        z0 = a0 * z_r[k - 1] + b0 * z_r[k]
        z1 = a1 * z_r[k] + b1 * z_r[k + 1]
        f0 = a0 * F[k - 1] + b0 * F[k]
        f1 = a1 * F[k] + b1 * F[k + 1]
        # First compute twice the integral
        A = (z_r[k] - z0) * (f0 + F[k]) + (z1 - z_r[k]) * (F[k] + f1)
        A = 0.5 * A / (z1 - z0)  # The average
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), A)

    def test_surface(self):
        """Correct for z_[-1] < z0 < z1"""
        N = 10
        z_r = np.linspace(-95, -5, N)
        F = np.random.random(z_r.shape)
        z0 = z_r[-1] + 7.2
        z1 = z_r[-1] + 13.0
        A = F[-1]
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), A)

    def test_bottom(self):
        """Correct for z0 < z1 < z_r[0]"""
        N = 10
        z_r = np.linspace(-95, -5, N)
        F = np.random.random(z_r.shape)
        z0 = z_r[0] - 27.2
        z1 = z_r[0] - 13.0
        A = F[0]
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), A)

    def test_near_surface(self):
        """Correct for z_r_[-2] < z0 < z_r_[-1] < z1"""
        N = 10
        z_r = np.linspace(-95, -5, N)
        F = np.random.random(z_r.shape)
        a0, b0 = 0.3, 0.7  # sum = 1
        z0 = a0 * z_r[-2] + b0 * z_r[-1]
        z1 = z_r[-1] + 13.0
        f0 = a0 * F[-2] + b0 * F[-1]
        f1 = F[-1]
        A = (z_r[-1] - z0) * (f0 + F[-1]) + 2 * (z1 - z_r[-1]) * f1
        A = 0.5 * A / (z1 - z0)
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), A)

    def test_near_bottom(self):
        """Correct for z0 < z_r[0] < z1 < z_r[1]"""
        N = 10
        z_r = np.linspace(-95, -5, N)
        F = np.random.random(z_r.shape)
        a1, b1 = 0.6, 0.4  # sum = 1
        z0 = z_r[0] - 11.4
        z1 = a1 * z_r[0] + b1 * z_r[1]
        f0 = F[0]
        f1 = a1 * F[0] + b1 * F[1]
        A = 2 * (z_r[0] - z0) * f0 + (z1 - z_r[0]) * (f0 + f1)
        A = 0.5 * A / (z1 - z0)
        self.assertAlmostEqual(z_average(F, z_r, z0, z1), A)

    # ta inn masse tester med shape

    # ....
    # OBS, raiser ValueError for feil grunn
    def test_order(self):
        """Raise ValueError if z1 >= z2"""
        z_r = np.linspace(-95, -5, 10)
        F = np.random.random(z_r.shape)
        z0, z1 = -11.3, -45.0
        self.assertRaises(ValueError, z_average, F, z_r, z0, z1)


# --------------------------------------


class test_z_average(unittest.TestCase):
    def test_constant(self):
        """If constant, returns the constant"""
        F0 = 5.0
        z_r = np.linspace(-95, -4, 10)
        F = F0 + np.zeros_like(z_r)
        z0 = -52.7
        z1 = -11.36
        #        F1 = z_average2(F, z_r, z0, z1)
        F1 = z_average(F, z_r, z0, z1)
        self.assertAlmostEqual(F1, F0)

    def test_linear(self):
        """If linear, returns the average of the end points"""
        z_r = np.linspace(-95, -4, 10)
        F = z_r.copy()
        z0 = -52.7
        z1 = -11.36
        F0 = 0.5 * (z0 + z1)
        F1 = z_average(F, z_r, z0, z1)
        self.assertAlmostEqual(F1, F0)


# --------------------------------------

if __name__ == "__main__":
    unittest.main()

import sys
import unittest

import numpy as np

from roppy.sample import sample2D

sys.path = ["..", *sys.path]


class Test_sample2D(unittest.TestCase):
    def test_arguments(self):
        # Make a grid for testing
        imax, jmax = 10, 7
        A = np.zeros((jmax, imax))

        # Scalars
        X, Y = 2.7, 3.5
        self.assertTrue(np.isscalar(sample2D(A, X, Y)))

        # Conformal arrays
        X = np.array([1, 2])
        Y = np.array([3, 4])
        B = sample2D(A, X, Y)
        self.assertEqual(B.shape, (2,))

        # Conformal arrays 2
        X = np.arange(6).reshape(2, 3)
        Y = np.array([1, 2]).reshape(2, 1)
        B = sample2D(A, X, Y)
        self.assertEqual(B.shape, (2, 3))

        # Scalar and array
        X = 4
        Y = np.array([1, 2])
        B = sample2D(A, X, Y)
        self.assertEqual(B.shape, (2,))

        # Nonconformal arrays
        X = np.arange(6).reshape(2, 3)
        Y = np.array([1, 2])
        self.assertRaises(ValueError, sample2D, A, X, Y)

        # 1D sequences such as lists
        X = [1.1, 2.2, 3.3]
        Y = [3.5]
        B = sample2D(A, X, Y)
        self.assertEqual(B.shape, (3,))

    # ---------------------------------------

    def test_boundary(self):
        """Works correctly close to boundary"""
        imax, jmax = 10, 7
        A = np.ones((jmax, imax))
        x, y = 0.2, 5.9
        self.assertEqual(sample2D(A, x, y), 1)

    def test_outside(self):
        """Handle outside values correctly"""
        imax, jmax = 10, 7
        A = np.zeros((jmax, imax))
        x, y = 9.2, 4
        self.assertRaises(ValueError, sample2D, A, x, y)
        b = sample2D(A, x, y, outside_value=np.nan)
        self.assertTrue(np.isnan(b))

    def test_bilinear(self):
        """Exact for bilinear function"""

        def f(x, y):
            return 3.0 + 2 * x + 1.4 * y + 0.2 * x * y

        imax, jmax = 10, 7
        JJ, II = np.meshgrid(np.arange(jmax), np.arange(imax))
        A = f(JJ, II)
        X, Y = 5.2, 4.1
        Z = f(X, Y)
        B = sample2D(A, X, Y)
        self.assertEqual(B, Z)

    def test_nodes(self):
        """Exact at nodes"""

        def f(x, y):
            return x**2 + y**2

        imax, jmax = 10, 7
        JJ, II = np.meshgrid(np.arange(jmax), np.arange(imax))
        A = f(JJ, II)
        i, j = 4, 3
        B = sample2D(A, i, j)
        self.assertEqual(B, A[j, i])

    # Use this landmask (origon=lower left)
    #    | | |x|x|
    #    |x| |x|x|
    #    |x| | | |
    #    | | | | |
    #
    def test_masked(self):
        """Interpolates correctly in the presence of mask"""
        A = np.array(
            [
                [1, 1, 1, 1],  # j=0
                [0, 1, 1, 1],  # j=1
                [0, 1, 0, 0],  # j=2
                [1, 1, 0, 0],
            ]
        )  # j=3
        M = A

        # sea point outside halo
        x, y = 1.8, 0.8
        self.assertEqual(sample2D(A, x, y), 1)
        self.assertEqual(sample2D(A, x, y, mask=M), 1)

        # sea point in halo
        x, y = 1.8, 1.3
        self.assertAlmostEqual(sample2D(A, x, y), 1 - 0.8 * 0.3)
        self.assertEqual(sample2D(A, x, y, mask=M), 1)

        # land point in halo
        x, y = 1.8, 1.8
        b_unmask = sample2D(A, x, y, undef_value=np.nan)
        self.assertAlmostEqual(b_unmask, 1 - 0.8 * 0.8)
        b_mask = sample2D(A, x, y, mask=M, undef_value=np.nan)
        self.assertEqual(b_mask, 1)

        # land point outside halo
        x, y = 2.8, 2.4
        b_unmask = sample2D(A, x, y, undef_value=np.nan)
        self.assertEqual(b_unmask, 0)
        b_mask = sample2D(A, x, y, mask=M, undef_value=np.nan)
        self.assertTrue(np.isnan(b_mask))

    # ------------------------------------------------

    def test_offset_forwards(self):
        # Make a grid with a bilinear function
        def f(x, y):
            return 3.0 + 2 * x + 1.4 * y + 0.2 * x * y

        # f = lambda y, x: x

        # Make a grid with offset
        i0, i1, j0, j1 = 2, 10, 3, 8
        II0, JJ0 = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1))

        A = f(II0, JJ0)
        x, y = 8.9, 4.1
        z = f(x, y)

        a = sample2D(A, x - i0, y - j0)
        self.assertAlmostEqual(a, z)


if __name__ == "__main__":
    unittest.main()

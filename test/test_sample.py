# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys
sys.path = [".."] + sys.path
#print sys.path
from roppy.sample import sample2D


class test_sample2D(unittest.TestCase):

    # Mangler test for list

    def test_arguments(self):
        # Make a grid for testing
        imax, jmax = 10, 7
        A = np.zeros((jmax, imax))
        
        # Scalars
        X, Y = 2.7, 3.5
        self.assertTrue(np.isscalar(sample2D(A, X, Y)))
        
        # Conformal arrays
        X = np.array([1,2])
        Y = np.array([3,4])
        B = sample2D(A, X, Y)
        self.assertEqual(B.shape, (2,))

        # Conformal arrays 2
        X = np.arange(6).reshape(2,3)
        Y = np.array([1,2]).reshape(2,1)
        B = sample2D(A, X, Y)
        self.assertEqual(B.shape, (2,3))

        # Scalar and array
        X = 4
        Y = np.array([1,2])
        B = sample2D(A, X, Y)
        self.assertEqual(B.shape, (2,))

        # Nonconformal arrays 
        X = np.arange(6).reshape(2,3)
        Y = np.array([1,2])
        self.assertRaises(ValueError, sample2D, A, X, Y)

    def test_boundary(self):
        imax, jmax = 10, 7
        A = np.ones((jmax, imax))
        x, y = 0.2, 5.9
        self.assertEqual(sample2D(A, x, y), 1)

    def test_outside(self):
        imax, jmax = 10, 7
        A = np.zeros((jmax, imax))
        x, y = 10.2, 4
        self.assertRaises(ValueError, sample2D, A, x, y)
        b = sample2D(A, x, y, outside_value=np.nan)
        self.assertTrue(np.isnan(b))

    def test_bilinear(self):
        """Exact for bilinear function"""
        f = lambda x, y: 3.0 + 2*x + 1.4*y + 0.2*x*y
        imax, jmax = 10, 7
        JJ, II = np.meshgrid(np.arange(jmax), np.arange(imax))
        A = f(JJ, II)
        X, Y = 5.2, 4.1
        Z = f(X, Y)
        B = sample2D(A, X, Y)
        self.assertEqual(B, Z)

    def test_nodes(self):
        """Exact at nodes"""
        f = lambda x, y : x**2 + y**2
        imax, jmax = 10, 7
        JJ, II = np.meshgrid(np.arange(jmax), np.arange(imax))
        A = f(JJ, II)
        i, j = 4, 3
        B = sample2D(A, i, j)
        self.assertEqual(B, A[j,i])

    # Tegn opp og sjekk
    def test_masked(self):
        jmax, imax = 3, 3
        A = np.array([[0,0,1],
                      [0,1,0],
                      [1,1,0]])
        M = A
        # sea point outside halo
        x, y = 0.2, 0.2
        self.assertEqual(sample2D(A, x, y), 1)
        self.assertEqual(sample2D(A, x, y, mask=M), 1)
        
        
        
        
if __name__ == '__main__':
    unittest.main()

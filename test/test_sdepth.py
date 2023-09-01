"""Unit tests for the vertical depth functions for ROMS grid"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# ----------------------------------

import numpy as np

from roppy import s_stretch, sdepth

# ------------------------------------

"""Setup unstretched and stretched vertical scales"""
N = 10
# Make N uniformly distributed values from -1+0.5/N to -0.5/N
S = -1.0 + (0.5 + np.arange(N)) / N
# Make a random increasing sequence of N values between -1 to 0
rng = np.random.default_rng()
C_random = rng.uniform(-1, 0, N)
C_random.sort()

# --------------------------
# Test the sdepth function
# --------------------------

# --- Default = Vtransform=1


def test1D_unstretched():
    """C = S should give equally distributed sigma-coordinates"""
    H = 100.0
    Hc = 10.0
    C = S
    Z = sdepth(H, Hc, C)
    assert np.allclose(Z, H * C)


def test_hc_is_zero():
    """Hc = 0 should give only the stretched values"""
    H = 100.0
    Hc = 0.0
    C = C_random
    Z = sdepth(H, Hc, C)
    assert np.allclose(Z, H * C)


def test_hc_is_h():
    """Hc = H should remove effect of stretching"""
    H = 100.0
    Hc = H
    C = C_random
    Z = sdepth(H, Hc, C)
    assert np.allclose(Z, H * S)


def test_shape():
    # Test that the shape comes out correctly
    H = np.array([[100.0, 90.0, 80.0], [70.0, 70.0, 60.0]])
    Hc = 10.0
    C = C_random
    Z = sdepth(H, Hc, C)
    Mp, Lp = H.shape
    assert Z.shape == (len(C), Mp, Lp)


# --- Vtransform=2


# class test_sdepth_Vtransform2(unittest.TestCase):
def test1D_unstretched_Vtransform2():
    H = 100.0
    Hc = 10.0
    N = 10
    C = -1.0 + (0.5 + np.arange(N)) / N  # equally distributed
    Z = sdepth(H, Hc, C, Vtransform=2)
    # Z should now be equally distributed
    # Z = A = array([-95.0,-85.0, ..., -5.0])
    A = np.arange(-95.0, -4.0, 10.0)
    assert np.allclose(Z, A)


def test_hc_is_zero_Vtransform2():
    # With Hc = 0, just scaled sc_r scaled by H
    H = 100.0
    Hc = 0.0
    N = 10
    cs_r = (-1.0 + (0.5 + np.arange(N)) / N) ** 3
    Z = sdepth(H, Hc, cs_r, Vtransform=2)
    A = H * cs_r
    assert np.allclose(Z, A)


def test_hc_is_h_Vtransform2():
    # With Hc = H, mean of stretched and unstretched
    H = 100.0
    Hc = H
    N = 10
    S = -1.0 + (0.5 + np.arange(N)) / N
    C = S**3
    Z = sdepth(H, Hc, C, Vtransform=2)
    assert np.allclose(Z, 0.5 * H * (S + C))


def test_shape_Vtransform2():
    # Test that the shape comes out correctly
    H = np.array([[100.0, 90.0, 80.0], [70.0, 70.0, 60.0]])
    Hc = 10.0
    N = 10
    S = -1.0 + (0.5 + np.arange(N)) / N
    C = S**3
    Z = sdepth(H, Hc, C, Vtransform=2)
    Mp, Lp = H.shape
    assert Z.shape == (N, Mp, Lp)


# -------------
# s_stretch
# -------------

# --- Default = Vstretching=1


def test_valid_output():
    """For rho-points C should obey -1 < C[i] < C[i+1] < 0"""
    N = 30
    theta_s = 6.0
    theta_b = 0.6
    C = s_stretch(N, theta_s, theta_b)
    Cp = np.concatenate(([-1], C, [0]))
    D = np.diff(Cp)
    assert all(D > 0)
    assert len(C) == N


def test_valid_output_w():
    """For w-points C should obey -1 <= C[i] < C[i+1] <= 0"""
    N = 30
    theta_s = 6.0
    theta_b = 0.6
    C = s_stretch(N, theta_s, theta_b, stagger="w")
    # Check increasing
    assert all(np.diff(C) > 0)
    # End points
    assert C[0] == -1.0
    assert C[-1] == 0.0
    # Length
    assert len(C) == N + 1


# --- Vstretching=4


def test_valid_output_stretch4():
    """For rho-points C should obey -1 < C[i] < C[i+1] < 0"""
    N = 30
    theta_s = 6.0
    theta_b = 0.6
    C = s_stretch(N, theta_s, theta_b, Vstretching=4)
    Cp = np.concatenate(([-1], C, [0]))
    D = np.diff(Cp)
    assert all(D > 0)
    assert len(C) == N


def test_valid_output_w_stretch4():
    """For w-points, C should obey -1 = C[0] < C[i] < C[i+1] < C[N] = 0"""
    N = 30
    theta_s = 6.0
    theta_b = 0.6
    C = s_stretch(N, theta_s, theta_b, Vstretching=4, stagger="w")
    # Check increasing
    assert all(np.diff(C) > 0)
    # End points
    assert C[0] == -1.0
    assert C[-1] == 0.0
    # Length
    assert len(C) == N + 1

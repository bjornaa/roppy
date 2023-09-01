import numpy as np
import pytest

from roppy import zslice

# Testing the zslice function
rng = np.random.default_rng()


def test_constant():
    """Slice of constant field returns the constant"""
    z_r = np.linspace(-95, -5, 10)

    const = 3.14
    F = np.zeros_like(z_r) + const
    z = -52.0
    assert np.allclose(zslice(F, z_r, z), const)
    z = -52  # Test integer depth
    assert np.allclose(zslice(F, z_r, z), const)


def test_linearity():
    """zslice is linear in the field F"""
    a, b = 3.8, 11.2
    z_r = np.linspace(-95, -5, 10)
    F1 = np.random.random(z_r.shape)
    F2 = np.random.random(z_r.shape)
    F = a * F1 + b * F2
    z = -52.0
    v = zslice(F, z_r, z)
    v1 = zslice(F1, z_r, z)
    v2 = zslice(F2, z_r, z)
    assert np.allclose(v, a * v1 + b * v2)


def test_identity():
    """zslice returns z if F == z_r"""
    z_r = np.linspace(-95, -5, 10)
    F = np.linspace(-95, -5, 10)
    z = -52.0
    assert zslice(F, z_r, z) == z


def test_interpolate():
    """zslice interpolates, z in z_r gives F-value"""
    z_r = np.linspace(-95, -5, 10)
    F = np.random.random(z_r.shape)
    k = 4
    z = z_r[k]
    assert np.allclose(zslice(F, z_r, z), F[k])


def test_linear_interpolation():
    """The interpolation is linear"""
    z_r = np.linspace(-95, -5, 10)
    F = np.random.random(z_r.shape)
    a, b = 0.3, 0.7  # sum = 1
    k = 4
    z = a * z_r[k - 1] + b * z_r[k]
    v = a * F[k - 1] + b * F[k]
    assert np.allclose(zslice(F, z_r, z), v)


def test_extrapolate():
    """Extrapolation should extend constantly outside highest and lowest values"""
    z_r = np.linspace(-95, -5, 10)
    F = np.random.random(z_r.shape)
    assert zslice(F, z_r, -2) == F[-1]  # high value
    assert zslice(F, z_r, -122.3) == F[0]  # low value


def test_interp():
    """zslice of 1D field is np.interp"""
    z_r = np.linspace(-95, -5, 10)
    F = np.random.random(z_r.shape)
    z = -52.0
    assert np.allclose(zslice(F, z_r, z), np.interp(z, z_r, F))


def test_non_equidistant():
    """zslice should work with non-equidistand z_r"""
    # Make a random increasing sequence z_r between -100 and 0
    z_r = -100 * rng.random(10)
    z_r.sort()
    F = rng.random(10)
    z = -52.0
    assert np.allclose(zslice(F, z_r, z), np.interp(z, z_r, F))


def test_shapes_OK():
    """Correct input shapes gives correct output shape"""

    # Make a z_r array of shape (N, Mp, Lp)
    # in this case (10,3,2)
    K = np.linspace(-95, -5, 10)
    z_r = np.transpose([[K, 2 * K, K], [K, 0.7 * K, 1.2 * K]])
    # F has same shape
    F = np.random.random(z_r.shape)
    # Scalar z
    z = -52.0
    assert zslice(F, z_r, z).shape == z_r.shape[1:]
    # Horizontal z with shape z_r.shape[1:] = (Mp, Lp)
    z = np.array([-10.2, -14.0, -28, -2.4, -88.7, -122])
    z.shape = (3, 2)  # z has correct shape
    assert zslice(F, z_r, z).shape == z_r.shape[1:]


def test_Fshape_wrong():
    """Raise exception if F.shape != z_r.shape"""

    # Make a z_r array of shape (N, Mp, Lp)
    # in this case (10,3,2)
    K = np.linspace(-95, -5, 10)
    z_r = np.transpose([[K, 2 * K, K], [K, 0.7 * K, 1.2 * K]])
    # F has same shape
    F = np.random.random(z_r.shape)
    F.shape = (10, 2, 3)  # Give F wrong shape
    z = -52.0
    with pytest.raises(ValueError):
        zslice(F, z_r, z)


def test_zshape_wrong():
    """Raise exception if z.shape is wrong"""

    # Make a z_r array of shape (N, Mp, Lp)
    # in this case (10,3,2)
    K = np.linspace(-95, -5, 10)
    z_r = np.transpose([[K, 2 * K, K], [K, 0.7 * K, 1.2 * K]])
    # F has same shape
    F = np.random.random(z_r.shape)
    z = np.array([-10.2, -14.0, -28, -2.4, -88.7, -122])
    z.shape = (2, 3)  # z is 2D, correct size, wrong shape
    with pytest.raises(ValueError):
        zslice(F, z_r, z)


def rest_array_values():
    """Test that interpolation gives correct results with array input"""
    # Make a z_r array of shape (N, Mp, Lp)
    # in this case (10,3,2)
    K = np.linspace(-95, -5, 10)
    z_r = np.transpose([[K, 2 * K, K], [K, 0.7 * K, 1.2 * K]])
    # F has same shape
    F = np.random.random(z_r.shape)
    j, i = 2, 1
    z = -52.0  # z is scalar
    Fz = zslice(F, z_r, z)
    assert np.allclose(Fz[j, i], np.interp(z, z_r[:, j, i], F[:, j, i]))

    z = np.array([-10.2, -14.0, -28, -2.4, -88.7, -122])
    z.shape = (3, 2)  # z has correct shape
    Fz = zslice(F, z_r, z)
    assert np.allclose(Fz[j, i], np.interp(z[j, i], z_r[:, j, i], F[:, j, i]))

import re

import numpy as np
import pytest
from scipy.special import jacobi

from lumicks.pyoptics.mathutils.zernike import _jacobi, zernike


def piston(r):
    return np.ones_like(r)


def tilt(r):
    return 2 * r


def defocus(r):
    return 3**0.5 * (2 * r**2 - 1)


def astigmatism(r):
    return 6**0.5 * r**2


def coma(r):
    return 8**0.5 * (3 * r**3 - 2 * r)


def spherical(r):
    return 5**0.5 * (6 * r**4 - 6 * r**2 + 1)


def secondary_astigmatism(r):
    return 10**0.5 * (4 * r**4 - 3 * r**2)


def test_zernike_raises():
    # rules:
    # 1. n, m ∈ N
    # 1. n, m > 0
    # 2. m <= n
    # 3. if n is odd, m has to be odd, and v.v.
    # 4. if n is odd, m > 0
    r = np.asarray(0.0)
    for n, m in ((3.0, 1), (3, 1.0)):
        with pytest.raises(ValueError, match="n and m need to be integers"):
            zernike(n, m, r)  # type: ignore
    for n, m in ((3, -1), (-3, 1), (3, 4)):
        with pytest.raises(
            ValueError,
            match="n and m need to be larger than zero, and m needs to be less than or equal to n",
        ):
            zernike(n, m, r)
    for n, m in ((4, 3), (3, 2)):
        with pytest.raises(
            ValueError, match=re.escape("If n is odd (even), m has to be odd (even)")
        ):
            zernike(n, m, r)
    with pytest.raises(ValueError, match="The value of m cannot be zero if n is odd"):
        zernike(3, 0, r)


def test_zernike_explicit():
    """Test against lower-order explicit Zernike polynomials (radial part)"""
    r = np.linspace(0, 1, 50)
    np.testing.assert_allclose(zernike(0, 0, r), piston(r))
    np.testing.assert_allclose(zernike(1, 1, r), tilt(r))
    np.testing.assert_allclose(zernike(2, 0, r), defocus(r))
    np.testing.assert_allclose(zernike(2, 2, r), astigmatism(r))
    np.testing.assert_allclose(zernike(3, 1, r), coma(r))
    np.testing.assert_allclose(zernike(4, 0, r), spherical(r))
    np.testing.assert_allclose(zernike(4, 2, r), secondary_astigmatism(r))


@pytest.mark.parametrize("n, alpha, beta", ((11, 0.5, 0.5), (32, 0.0, 0.0), (40, 1.0, 1.5)))
def test_jacobi(n: int, alpha: float, beta: float):
    x = np.linspace(-1, 1, 50)
    np.testing.assert_allclose(jacobi(n, alpha, beta)(x), _jacobi(n, alpha, beta, x))

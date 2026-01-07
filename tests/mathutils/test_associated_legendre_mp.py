import mpmath as mp
import numpy as np
import pytest

from lumicks.pyoptics.mathutils.associated_legendre import associated_legendre


def associated_legendre_mp(n: int, x: mp.mpc):
    """associated_legendre(n, x): Return the 1st order (m == 1) of the
    associated Legendre polynomial of degree n, evaluated at x [-1..1]

    Uses a Clenshaw recursive algorithm for the evaluation, specific for the
    1st order associated Legendre polynomials.

    See Clenshaw, C. W. (July 1955). "A note on the summation of Chebyshev
    series". Mathematical Tables and Other Aids to Computation. 9 (51): 118
    """

    def _fi1(x):
        """First order associated Legendre polynomial evaluated at x"""
        return -(((1 + x) * (1 - x)) ** mp.mpf("0.5"))

    def _fi2(x):
        """Second order associated Legendre polynomial evaluated at x"""
        return -3 * x * ((1 + x) * (1 - x)) ** mp.mpf("0.5")

    if n == 1:
        return _fi1(x)
    if n == 2:
        return _fi2(x)
    bk2 = mp.mpc(0 + 0j)
    bk1 = mp.mpc(1 + 0j)
    for k_ in range(n - 1, 1, -1):
        k = mp.mpf(k_)
        bk = ((2 * k + 1) / k) * x * bk1 - (k + 2) / (k + 1) * bk2
        bk2 = bk1
        bk1 = bk
    return _fi2(x) * bk1 - mp.mpf("1.5") * _fi1(x) * bk2


@pytest.mark.parametrize("order", range(200, 300, 3))
@pytest.mark.parametrize("check_complex", (True, False))
def test_legendre(order, check_complex):
    """
    Test numpy float64 implementation of associated_legendre against a
    higher-precision software floating point format provided by mpmath.
    It should effectively be about a float192 (mp.dps = 45)
    """
    mp.mp.dps = 45
    x = np.linspace(-1, 1, 101)
    if check_complex:
        x = x + 0.1j
    y1 = associated_legendre(order, x)
    y2 = []
    for point in x:
        y2.append(complex(associated_legendre_mp(order, point)))
    np.testing.assert_allclose(y1, y2, rtol=1e-11)

import numpy as np
import numpy.polynomial as npp
import pytest

from lumicks.pyoptics.mathutils.associated_legendre import (
    associated_legendre,
    associated_legendre_dtheta,
    associated_legendre_over_sin_theta,
)


def associated_legendre_npp(n, x):
    """Return the 1st order (m == 1) of the
    associated Legendre polynomial of degree n, evaluated at x [-1..1]

    This is a reference implementation in numpy, which is more generic and therefore slower.
    Furthermore, it's not compatible with Numba.
    """
    orders = np.zeros(n + 1)
    orders[n] = 1
    c = npp.Legendre(npp.legendre.legder(orders))
    return -c(x) * np.sqrt(1 - x**2)


def _associated_legendre_dtheta(n: int, cos_theta: np.ndarray, alp_sin_pre=(None, None)):
    """Evaluate the derivative of the associated legendre polynomial
    :math:`dP_n^1(\\cos(\\theta))/d\\theta`.

    This is a previous implementation that has been replaced by a vectorized version for actual use

    Parameters
    ----------
    n : int
        degree
    cos_theta : np.ndarray
        The values of :math:`\\cos(\\theta)` to evaluate :math:`dP_n^1(\\cos(\\theta))/d\\theta`.
    alp_sin_pre : tuple, optional
        If :math:`P_n^1(\\cos(\\theta))/\\sin(\\theta)` are precalculated for order `n` and order `n -
        1`, then these can be provided to the function, which saves processing time. If these are
        not available, they will be calculated on the fly. By default the value is (None, None)

    Returns
    -------
    np.ndarray
        The derivative at :math:`\\cos(\\theta)`.
    """
    if n == 1:
        return -cos_theta

    if alp_sin_pre[0] is None or alp_sin_pre[1] is None:
        alp = associated_legendre_over_sin_theta(n, cos_theta)
        alp_1 = associated_legendre_over_sin_theta(n - 1, cos_theta)
    else:
        alp = alp_sin_pre[0]
        alp_1 = alp_sin_pre[1]

    # See https://dlmf.nist.gov/14.10 eq. 14.10.5:
    # (1−x^2) ⁢d𝖯_n^μ⁡(x) / dx = (n + μ) ⁢𝖯_{n − 1}^μ⁡(x) − n ⁢x ⁢𝖯_n^μ⁡(x).
    # For µ = 1 and x = cos(theta):
    # d𝖯_n⁡(cos θ) / d(cos θ) = [(n + 1) ⁢𝖯_{n − 1}⁡(cos θ) − n cos θ ⁢𝖯_n⁡(cos θ)]/sin^2(θ)

    # Required is d𝖯_n⁡(cos θ) / dθ, which is d𝖯_n⁡(cos θ) / d(cos θ) * d(cos θ)/dθ
    # = - d𝖯_n⁡(cos θ) / d(cos θ) sin(θ)
    # = [n cos θ ⁢𝖯_n⁡(cos θ) - (n + 1) ⁢𝖯_{n − 1}⁡(cos θ)]/sin(θ)
    return n * cos_theta * alp - (n + 1) * alp_1


@pytest.mark.parametrize("order", range(1, 200))
@pytest.mark.parametrize("check_complex", (True, False))
def test_legendre(order, check_complex):
    x = np.linspace(-1, 1, 101)
    if check_complex:
        x = x + 0.1j
    y1 = associated_legendre(order, x)
    y2 = associated_legendre_npp(order, x)
    np.testing.assert_allclose(y1, y2)


@pytest.mark.parametrize("max_order", (1, 10, 100))
def test_legendre_dtheta(max_order):
    x = np.linspace(-1, 1, 101)
    alp_sin = np.zeros((max_order, x.size))
    alp_dtheta = np.empty_like(alp_sin)
    alp_dtheta_ref = np.empty_like(alp_sin)

    for order in range(1, max_order + 1):
        alp_sin[order - 1, :] = associated_legendre_over_sin_theta(order, x)
        alp_pre = (alp_sin[order - 1], alp_sin[order - 2]) if order > 1 else (None, None)
        alp_dtheta_ref[order - 1, :] = _associated_legendre_dtheta(order, x, alp_pre)

    associated_legendre_dtheta(x, alp_sin, alp_dtheta)

    np.testing.assert_allclose(alp_dtheta_ref, alp_dtheta)

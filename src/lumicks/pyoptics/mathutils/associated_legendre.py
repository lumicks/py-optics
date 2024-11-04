"""Associated Legendre polynomials"""

import numpy as np
from numba import njit


@njit(cache=True, parallel=False, fastmath=False)
def associated_legendre(n: int, x: np.ndarray):
    """Return :math:`P^1_n(x)`, the 1st order (m == 1) of the
    associated Legendre polynomial of degree n , evaluated at x [-1..1]. Uses the result of
    `associated_legendre_over_sin_theta()`, multiplied by :math:`\\sqrt(1 - x^2)`

    Parameters
    ----------
    n : int
        degree
    x : np.ndarray
        array of values to calculate :math:`P^1_n(x)`. Commonly, `x` is also written as
        :math:`\\cos(\\theta)`

    Returns
    -------
    np.ndarray
        The values of :math:`P^1_n(x)` for `n` at `x`.
    """
    sqrt_x = ((1 + x) * (1 - x)) ** 0.5
    if n == 1:
        return -sqrt_x
    return associated_legendre_over_sin_theta(n, x) * sqrt_x


@njit(cache=True, parallel=False)
def associated_legendre_dtheta(cos_theta: np.ndarray, alp_sin_pre, out: np.ndarray):
    """Evaluate the derivative of the associated legendre polynomial
    :math:`dP_n^1(\\cos(\\theta))/d\\theta`.

    Parameters
    ----------
    n : int
        degree
    cos_theta : np.ndarray
        The values of :math:`\\cos(\\theta)` to evaluate :math:`dP_n^1(\\cos(\\theta))/d\\theta`.
    alp_sin_pre : np.ndarray
        An (n, len(cos_theta)) Numpy array of :math:`P_n^1(\\cos(\\theta))/\\sin(\\theta)` values
        that are precalculated for up to order `n`.

    Returns
    -------
    np.ndarray
        The derivative to :math:`\\theta` evaluated at :math:`\\cos(\\theta)`.
    """
    # See https://dlmf.nist.gov/14.10 eq. 14.10.5:
    # (1−x^2) ⁢d𝖯_n^μ⁡(x) / dx = (n + μ) ⁢𝖯_{n − 1}^μ⁡(x) − n ⁢x ⁢𝖯_n^μ⁡(x).
    # For µ = 1 and x = cos(theta):
    # d𝖯_n⁡(cos θ) / d(cos θ) = [(n + 1) ⁢𝖯_{n − 1}⁡(cos θ) − n cos θ ⁢𝖯_n⁡(cos θ)]/sin^2(θ)

    # Required is d𝖯_n⁡(cos θ) / dθ, which is d𝖯_n⁡(cos θ) / d(cos θ) * d(cos θ)/dθ
    # = - d𝖯_n⁡(cos θ) / d(cos θ) sin(θ)
    # = [n cos θ ⁢𝖯_n⁡(cos θ) - (n + 1) ⁢𝖯_{n − 1}⁡(cos θ)]/sin(θ)

    cos_theta = cos_theta.reshape(1, cos_theta.size)
    out[0, :] = -cos_theta
    if alp_sin_pre.shape[0] == 1:
        return
    n = np.arange(2, stop=alp_sin_pre.shape[0] + 1).reshape(alp_sin_pre.shape[0] - 1, -1)

    out[1:, :] = n * alp_sin_pre[1:, :] * cos_theta - (n + 1) * alp_sin_pre[:-1, :]


@njit(cache=True, parallel=False, fastmath=False)
def associated_legendre_over_sin_theta(n, cos_theta):
    """Evaluate the associated legendre polynomial of order 1 and degree n,
    divided by sin(theta): :math:`dP_n^1(\\cos(\\theta))/\\sin(\\theta)`.

    Parameters
    ----------
    n : int
        Degree to calculate
    cos_theta : np.ndarray
        Cosine of the angles to calculate at.

    Returns
    -------
    np.ndarray
        Calculated values

    Notes
    -----
    The function calculates the following:

    .. math::
        P_n^1(\\cos(\\theta))/\\sin(\\theta)

    Uses a Clenshaw recursive algorithm for the evaluation, specific for the 1st order associated
    Legendre polynomials [1]_. This is numerically more stable that a direct evaluation of the
    polynomial, for high degrees of n


    ..  [1] Clenshaw, C. W. (July 1955). "A note on the summation of Chebyshev series". Mathematical
            Tables and Other Aids to Computation. 9 (51): 118
    """
    if n == 1:
        return -np.ones_like(cos_theta)
    if n == 2:
        return -3.0 * cos_theta
    bk2 = np.zeros(cos_theta.shape)
    bk1 = np.ones(cos_theta.shape)
    for k in range(n - 1, 1, -1):
        bk = ((2 * k + 1) / k) * cos_theta * bk1 - (k + 2) / (k + 1) * bk2
        bk2 = bk1
        bk1 = bk
    return -3.0 * (cos_theta * bk1 - bk2 / 2)

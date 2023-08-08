"""Associated Legendre polynomials"""

import numpy as np
from numba import njit


@njit(cache=True, parallel=False, fastmath=False)
def associated_legendre(n: int, x: np.ndarray):
    """associated_legendre(n, x): Return the 1st order (m == 1) of the
    associated Legendre polynomial of degree n, evaluated at x [-1..1]

    Uses the result of associated_legendre_over_sin_theta(), multiplied by sqrt(1 - x^2)
    """
    sqrt_x = ((1 + x) * (1 - x)) ** 0.5
    if n == 1:
        return -sqrt_x
    return associated_legendre_over_sin_theta(n, x) * sqrt_x


@njit(cache=True, parallel=False)
def associated_legendre_dtheta(n, cos_theta, alp_sin_pre=(None, None)):
    """
    Evaluate the derivative of the associated legendre polynomial
    P_n^1(cos(theta)), of order 1 and degree n, to the variable theta,
    where cos_theta == cos(theta).
    """
    if (alp_sin_pre[0] is None or alp_sin_pre[1] is None) and n > 1:
        alp = associated_legendre_over_sin_theta(n, cos_theta)
        alp_1 = associated_legendre_over_sin_theta(n - 1, cos_theta)
    else:
        alp = alp_sin_pre[0]
        alp_1 = alp_sin_pre[1]

    if n == 1:
        result = -cos_theta
    else:
        # See https://dlmf.nist.gov/14.10 eq. 14.10.5
        result = n * cos_theta * alp - (n + 1) * alp_1

    return result


@njit(cache=True, parallel=False, fastmath=False)
def associated_legendre_over_sin_theta(n, cos_theta):
    """Evaluate the associated legendre polynomial of order 1 and degree n,
    divided by sin(theta): P_n^1(cos(theta))/sin(theta)

    Uses a Clenshaw recursive algorithm for the evaluation, specific for the 1st order associated
    Legendre polynomials. This is numerically more stable that a direct evaluation of the
    polynomial, for high degrees of n

    See Clenshaw, C. W. (July 1955). "A note on the summation of Chebyshev series". Mathematical
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

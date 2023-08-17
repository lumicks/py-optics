"""Associated Legendre polynomials"""

import numpy as np
from numba import njit


@njit(cache=True, parallel=False, fastmath=False)
def associated_legendre(n: int, x: np.ndarray):
    """Return :math:`P^1_n(x)`, the 1st order (m == 1) of the
    associated Legendre polynomial of degree n , evaluated at x [-1..1]. Uses the result of
    `associated_legendre_over_sin_theta()`, multiplied by :math:`\sqrt(1 - x^2)`

    Parameters
    ----------
    n : int
        degree
    x : np.ndarray
        array of values to calculate :math:`P^1_n(x)`. Commonly, `x` is also written as
        :math:`\cos(\\theta)`

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
def associated_legendre_dtheta(n: int, cos_theta: np.ndarray, alp_sin_pre=(None, None)):
    """Evaluate the derivative of the associated legendre polynomial
    :math:`dP_n^1(\cos(\\theta))/d\\theta`.

    Parameters
    ----------
    n : int
        degree
    cos_theta : np.ndarray
        The values of :Math:`\cos(\\theta)` to evaluate :math:`dP_n^1(\cos(\\theta))/d\\theta`.
    alp_sin_pre : tuple, optional
        If :math:`P_n^1(\cos(\\theta))/\sin(\\theta)` are precalculated for order `n` and order `n -
        1`, then these can be provided to the function, which saves processing time. If these are
        not available, they will be calculated on the fly. By default the value is (None, None)

    Returns
    -------
    np.ndarray
        The derivative at :math:`\cos(\\theta)`.
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
    divided by sin(theta): :math:`dP_n^1(\cos(\\theta))/\sin(\\theta)`.

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
        P_n^1(\cos(\\theta))/\sin(\\theta)

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

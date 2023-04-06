"""Legendre polynomials"""

import numpy as np
import numpy.polynomial as npp
from numba import njit
from joblib import Memory

cachedir = 'miecalc_cache'
memory = Memory(cachedir, verbose=0)

@memory.cache
@njit(cache=True, parallel=True)
def associated_legendre(n: int, x: np.ndarray):
    """associated_legendre(n, x): Return the 1st order (m == 1) of the 
    associated Legendre polynomial of degree n, evaluated at x [-1..1]

    Uses a Clenshaw recursive algorithm for the evaluation, specific for the 
    1st order associated Legendre polynomials.

    See Clenshaw, C. W. (July 1955). "A note on the summation of Chebyshev 
    series". Mathematical Tables and Other Aids to Computation. 9 (51): 118
    """
    def _fi1(x):
        """First order associated Legendre polynomial evaluated at x"""
        return -(1-x**2)**0.5 

    def _fi2(x):
        """Second order associated Legendre polynomial evaluated at x"""
        return -3*x*(1-x**2)**0.5

    if n == 1:
        return _fi1(x)
    if n == 2:
        return _fi2(x)
    bk2 = np.zeros(x.shape)
    bk1 = np.ones(x.shape)
    for k in range(n - 1, 1, -1):
        bk = ((2 * k + 1) / k) * x * bk1 - (k + 2) / (k + 1) * bk2
        bk2 = bk1
        bk1 = bk
    return _fi2(x) * bk1 - 1.5 * _fi1(x) * bk2

def associated_legendre_npp(n, x):
    """associated_legendre(n, x): Return the 1st order (m == 1) of the 
    associated Legendre polynomial of degree n, evaluated at x [-1..1]

    This is a reference implementation in numpy, which is more generic and 
    therefore slower. Furthermore, it's not compatible with Numba.
    """
    orders = np.zeros(n + 1)
    orders[n] = 1
    c = npp.Legendre(npp.legendre.legder(orders))
    return -c(x) * np.sqrt(1 - x**2)


def associated_legendre_dtheta(n, cos_theta, alp_pre=(None, None)):
    """
    Evaluate the derivative of the associated legendre polynomial 
    P_n^1(cos(theta)), of order 1 and degree n, to the variable theta, 
    where cos_theta == cos(theta).
    """
    if alp_pre[0] is None:
        alp = associated_legendre(n, cos_theta)
        if n > 1:
            alp_1 = associated_legendre(n - 1, cos_theta)
    else:
        alp = alp_pre[0]
        alp_1 = alp_pre[1]

    with np.errstate(divide='ignore', invalid='ignore'):
        if n == 1:
            result = (n * cos_theta * alp) / np.sqrt(1 - cos_theta**2)
        else:
            # let it 'crash' on cos_theta == +/- 1, fix later
            # with np.errstate(divide='ignore', invalid='ignore'):
            result = (n * cos_theta * alp - (n + 1) * alp_1) / np.sqrt(1 -
                    cos_theta**2)

    result[cos_theta == 1] = -n * (n + 1) / 2
    result[cos_theta == -1] = -(-1)**n * n * (n + 1) / 2

    return result


def associated_legendre_over_sin_theta(n, cos_theta, alp_pre=None):
    """Evaluate the associated legendre polynomial of order 1 and degree n, 
    divided by sin(theta): P_n^1(cos(theta))/sin(theta)"""

    if alp_pre is None:
        alp = associated_legendre(n, cos_theta)
    else:
        alp = alp_pre

    # ignore division by zero, fix later
    with np.errstate(divide='ignore', invalid='ignore'):
        result = alp / np.sqrt(1 - cos_theta**2)

    result[cos_theta == 1] =  -n * (n + 1) / 2
    result[cos_theta == -1] = (-1)**n * n * (n + 1) / 2

    return result

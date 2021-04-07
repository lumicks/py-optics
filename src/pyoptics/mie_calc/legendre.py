"""Legendre polynomials"""

import numpy as np
import scipy.special as sp
import numpy.polynomial as npp


def associated_legendre(n, x):
    """Return the 1st order (m == 1) of the associated Legendre polynomial of degree n, evaluated at x"""
    orders = np.zeros(n+1)
    orders[n]=1
    c=npp.Legendre(npp.legendre.legder(orders))
    return -c(x) * np.sqrt(1 - x**2)


def associated_legendre_dtheta(n, cos_theta, alp_pre=(None, None)):
    """
    Evaluate the derivative of the associated legendre polynomial P_n^1(cos(theta)), of
    order 1 and degree n, to the variable theta, where cos_theta == cos(theta).
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
            with np.errstate(divide='ignore', invalid='ignore'):
                result = (n * cos_theta * alp - (n + 1) * alp_1) / np.sqrt(1 -
                        cos_theta**2)

    result[cos_theta == 1] = -n * (n + 1) / 2
    result[cos_theta == -1] = -(-1)**n * n * (n + 1) / 2

    return result


def associated_legendre_over_sin_theta(n, cos_theta, alp_pre=None):
    """Evaluate P_n^1(cos(theta))/sin(theta)"""

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

"""Clenshaw summation for Zernike polynomials"""

import numpy as np
from numba import njit

__all__ = ["zernike"]


def zernike(n: int, m: int, r: np.ndarray):
    """Return :math:`r^m P^(0,m)_n(2 r^2 - 1)`, the radial part of the Zernike polynomial of radial
    degree n and azimuthal frequency m.

    Parameters
    ----------
    n : int
        radial degree
    m : int
        azimuthal frequency
    r : np.ndarray
        array of values to evaluate the Zernike polynomial at

    Returns
    -------
    np.ndarray
        The values of :math:`r^m P^(0,m)_n(2 r^2 - 1)` for `n` and `m` at `r`, where
        :math:`P^(0,m)_n(x)` are the Jacobi polynomials. Multiply these results with cos(mθ) or
        sin(mθ) for m != 0 to get the full Zernike polynomial.

    Notes
    -----

    Uses a Clenshaw recursive algorithm for the evaluation of the Jacobi polynomials [1]_, [2]_. This is
    numerically more stable that a direct evaluation of the polynomial, for high degrees of n.


    ..  [1] Clenshaw, C. W. (July 1955). "A note on the summation of Chebyshev series". Mathematical
            Tables and Other Aids to Computation. 9 (51): 118
    ..  [2] https://dlmf.nist.gov/18.9 & https://en.wikipedia.org/wiki/Zernike_polynomials
    """
    if not all([isinstance(arg, int) for arg in (n, m)]):
        raise ValueError("n and m need to be integers")
    if n < 0 or m < 0 or m > n:
        raise ValueError(
            "n and m need to be larger than zero, and m needs to be less than or equal to n"
        )
    if m > 0 and ((n % 2 and not m % 2) or (m % 2 and not n % 2)):
        raise ValueError("If n is odd (even), m has to be odd (even)")
    if m == 0 and n % 2:
        raise ValueError("The value of m cannot be zero if n is odd")
    x = 2 * r**2 - 1
    normalization = (n + 1) ** 0.5 if m == 0 else (2 * (n + 1)) ** 0.5
    return r**m * _jacobi((n - m) // 2, 0, m, x) * normalization


@njit(cache=True, parallel=True)
def _jacobi(n: int, alpha: float, beta: float, x: np.ndarray):

    def _alpha(_n):
        if _n == 0 and alpha + beta in {0, -1}:
            return (0.5 * (alpha + beta) + 1) * x + 0.5 * (alpha - beta)
        return (
            (2 * _n + alpha + beta + 1) * (alpha**2 - beta**2)
            + (2 * _n + alpha + beta)
            * (2 * _n + alpha + beta + 1)
            * (2 * _n + alpha + beta + 2)
            * x
        ) / (2 * (_n + 1) * (_n + alpha + beta + 1) * (2 * _n + alpha + beta))

    def _beta(_n):
        return (
            -2
            * (_n + alpha)
            * (_n + beta)
            * (2 * _n + alpha + beta + 2)
            / (2 * (_n + 1) * (_n + alpha + beta + 1) * (2 * _n + alpha + beta))
        )

    y_k_2 = np.zeros_like(x)
    y_k_1 = np.ones_like(x)
    for k in range(n - 1, -1, -1):
        y_k = _alpha(k) * y_k_1 + _beta(k + 1) * y_k_2
        y_k_2 = y_k_1
        y_k_1 = y_k
    return y_k_1

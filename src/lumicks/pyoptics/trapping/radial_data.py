import numpy as np
import scipy.special as sp
from dataclasses import dataclass


@dataclass
class ExternalRadialData:
    """
    Data class that holds spherical Hankel functions evaluated at radii outside
    the bead, and related functions (derivatives)
    """

    k0r: np.ndarray
    krH: np.ndarray
    dkrH_dkr: np.ndarray


@dataclass
class InternalRadialData:
    """
    Data class that holds spherical Bessel functions evaluated at radii inside
    the bead, and related functions (derivatives)
    """

    sphBessel: np.ndarray
    jn_over_k1r: np.ndarray
    jn_1: np.ndarray


def calculate_external(k: float, radii: np.ndarray, n_orders: int):
    """
    Precompute the spherical Hankel functions and derivatives that only depend
    on the r coordinate. These functions will not change for any rotation of
    the coordinate system.
    """

    # Only calculate the spherical bessel function for unique values of k0r
    k0r = k * radii
    k0r_unique, inverse = np.unique(k0r, return_inverse=True)
    sqrt_x = np.sqrt(0.5 * np.pi / k0r_unique)
    krh_1 = np.sin(k0r) - 1j * np.cos(k0r)
    sphHankel = np.empty((n_orders, k0r.shape[0]), dtype=np.complex128)
    krH = np.empty(sphHankel.shape, dtype=np.complex128)
    dkrH_dkr = np.empty(sphHankel.shape, dtype=np.complex128)

    for L in range(1, n_orders + 1):
        sphHankel[L - 1, :] = (
            sqrt_x * (sp.jv(L + 0.5, k0r_unique) + 1j * sp.yv(L + 0.5, k0r_unique))
        )[inverse]
        krH[L - 1, :] = k0r * sphHankel[L - 1, :]
        # d/dp [p h_n(p)] = p h_n'(p) + h_n(p)
        # h_n'(p) = h_{n−1}⁡(p) − ((n + 1) / p) h_n⁡(p) (See https://dlmf.nist.gov/10.51 10.51.2)
        # Then, d/dp [p h_n(p)] = p h_{n−1}⁡(p) − (n + 1) h_n⁡(p) + h_n(p)
        #                       = p h_{n−1}⁡(p) − n h_n⁡(p)
        # With p = kr and n = L, we get:
        dkrH_dkr[L - 1, :] = krh_1 - L * sphHankel[L - 1, :]
        krh_1 = krH[L - 1, :]

    return ExternalRadialData(k0r, krH, dkrH_dkr)


def calculate_internal(k1: float, radii: np.ndarray, n_orders: int):
    """
    Precompute the spherical Bessel functions and related that only depend on
    the r coordinate, for the fields inside of the sphere.
    """

    k1r = k1 * radii
    k1r_unique, inverse = np.unique(k1r, return_inverse=True)
    sphBessel = np.zeros((n_orders, k1r.shape[0]), dtype="complex128")
    jn_over_k1r = np.zeros(sphBessel.shape, dtype="complex128")
    jn_1 = np.zeros(sphBessel.shape, dtype="complex128")
    jprev = np.empty(k1r.shape[0], dtype="complex128")
    jprev[k1r > 0] = np.sin(k1r[k1r > 0]) / k1r[k1r > 0]
    jprev[k1r == 0] = 1

    for L in range(1, n_orders + 1):
        sphBessel[L - 1, :] = sp.spherical_jn(L, k1r_unique)[inverse]
        jn_over_k1r[L - 1, k1r > 0] = sphBessel[L - 1, k1r > 0] / k1r[k1r > 0]
        jn_1[L - 1, :] = jprev
        jprev = sphBessel[L - 1, :]
    # The limit of the Spherical Bessel functions for jn(x)/x == 0, except
    # for n == 1. Then it is 1/3. See https://dlmf.nist.gov/10.52
    # For n > 1 taken care of by np.zeros(...)
    jn_over_k1r[0, k1r == 0] = 1 / 3

    return InternalRadialData(sphBessel, jn_over_k1r, jn_1)

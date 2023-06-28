import numpy as np
from numba import njit, prange
from .associated_legendre import (
    associated_legendre,
    associated_legendre_dtheta,
    associated_legendre_over_sin_theta
)


@njit(cache=True)
def _exec_lookup(source, lookup, r, c):
    """
    Compile the retrieval of the lookup table with Numba, as it's slightly
    faster
    """
    return source[:, lookup[r, c]]


class AssociatedLegendreData:
    __slots__ = (
        '_associated_legendre',
        '_associated_legendre_over_sin_theta',
        '_associated_legendre_dtheta',
        '_inverse'
    )

    def __init__(
        self,
        associated_legendre: np.ndarray,
        associated_legendre_over_sin_theta: np.ndarray,
        associated_legendre_dtheta: np.ndarray,
        inverse: np.ndarray,
    ) -> None:

        self._associated_legendre = associated_legendre
        self._associated_legendre_over_sin_theta = \
            associated_legendre_over_sin_theta
        self._associated_legendre_dtheta = associated_legendre_dtheta
        self._inverse = inverse

    def associated_legendre(self, r: int, c: int):
        return _exec_lookup(self._associated_legendre, self._inverse, r, c)

    def associated_legendre_over_sin_theta(self, r: int, c: int):
        return _exec_lookup(
            self._associated_legendre_over_sin_theta, self._inverse, r, c
        )

    def associated_legendre_dtheta(self, r: int, c: int):
        return _exec_lookup(
            self._associated_legendre_dtheta, self._inverse, r, c
        )


@njit(cache=True, parallel=True)
def _loop_over_rotations(
    local_coords: np.ndarray, radii: np.ndarray, aperture: np.ndarray,
    cos_theta: np.ndarray, sin_theta: np.ndarray,
    cos_phi: np.ndarray, sin_phi: np.ndarray
):
    """
    Loop over all possible rotations, in order to find the unique values of
    cos(theta) to reduce the computational load calculating Legendre
    polynomials.
    """
    rows, cols = np.nonzero(aperture)
    
    # Weird construct to work around tuple unpacking bug in Numba
    # https://github.com/numba/numba/issues/8772
    shape = (aperture.shape[0], aperture.shape[1], radii.size)
    cosTs = np.zeros(shape)

    index = radii == 0
    for r, c in zip(rows, cols):
        # this line is a workaround for a TypingError in Numba:
        ct = cosTs[r, c, :]

        # Rotate the coordinate system such that the x-polarization on the
        # bead coincides with theta polarization in global space
        # however, cos(theta) is the same for phi polarization!
        # A = (_R_th(cos_theta[p,m], sin_theta[p,m]) @
        #     _R_phi(cos_phi[p,m], -sin_phi[p,m]))
        # coords = A @ local_coords
        # z = coords[2, :]
        z = (
            local_coords[2, :] * cos_theta[r, c] - (
                local_coords[0, :] * cos_phi[r, c] +
                local_coords[1, :] * sin_phi[r, c]
            ) * sin_theta[r, c]
        )

        # Retrieve an array of all values of cos(theta)
        ct[:] = z / radii  # cos(theta)
        ct[index] = 1
        cosTs[r, c, :] = ct

    return cosTs


@njit(cache=True, parallel=True)
def _do_legendre_calc(
    cosT_unique: np.ndarray, abs_cosT_unique: np.ndarray,
    sign_inv: np.ndarray, sign_cosT_unique, n_orders: int
):
    """
    Numba compatible function that computes Associated Legendre Polynomials for
    n_order orders, plus derived quantities
    """

    # Allocate memory for the Associated Legendre Polynomials and derivatives
    alp = np.zeros((n_orders, sign_inv.size))
    alp_deriv = np.zeros_like(alp)
    alp_sin = np.zeros_like(alp)

    # Parity for order L: parity[0] == 1, for L == 1
    parity = (-1)**np.arange(n_orders)
    for L in prange(1, n_orders + 1):
        alp[L - 1, :] = associated_legendre(L, abs_cosT_unique)[sign_inv]
        if parity[L - 1] == -1:
            alp[L - 1, :] *= sign_cosT_unique
        alp_sin[L - 1, :] = associated_legendre_over_sin_theta(
            L, cosT_unique, alp[L - 1, :]
        )

    # For some reason, prange leads to numerical errors here
    for L in range(1, n_orders + 1):
        alp_prev = alp[L - 2, :] if L > 1 else None
        alp_deriv[L - 1, :] = associated_legendre_dtheta(
            L, cosT_unique, (alp[L - 1, :], alp_prev)
        )

    return alp, alp_sin, alp_deriv


def calculate_legendre(
    local_coords: np.ndarray, radii: np.ndarray, aperture: np.ndarray,
    cos_theta: np.ndarray, sin_theta: np.ndarray,
    cos_phi: np.ndarray, sin_phi: np.ndarray,
    n_orders: int
):
    """
    Find the value of cos(theta) for all coordinates (x, y, z) after rotating
    the coordinates along all possible directions, defined by the angles in the
    back focal plane.
    """
    if aperture.shape[0] != aperture.shape[1]:
        raise ValueError("Aperture matrix must be square")

    cosTs = _loop_over_rotations(
        local_coords, radii, aperture, cos_theta, sin_theta,
        cos_phi, sin_phi
    )

    shape = cosTs.shape
    cosTs = np.reshape(cosTs, cosTs.size)
    # rounding errors may make cos(theta) > 1 or < -1. Fix it to [-1..1]
    cosTs[cosTs > 1] = 1
    cosTs[cosTs < -1] = -1

    # Get the unique values of cos(theta) in the array
    cosT_unique, inverse = np.unique(cosTs, return_inverse=True)
    inverse = np.reshape(inverse, shape)

    # Get the signs of the cosines, and use parity to reduce the computational
    # load
    sign_cosT_unique = np.sign(cosT_unique)
    abs_cosT_unique, sign_inv = np.unique(
        np.abs(cosT_unique), return_inverse=True
    )

    alp, alp_sin, alp_deriv = _do_legendre_calc(
        cosT_unique, abs_cosT_unique, sign_inv, sign_cosT_unique, n_orders
    )
    return AssociatedLegendreData(alp, alp_sin, alp_deriv, inverse)

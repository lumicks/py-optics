import numpy as np
from numba import njit, prange
from dataclasses import fields
from .associated_legendre import associated_legendre_dtheta, associated_legendre_over_sin_theta
from .local_coordinates import CoordLocation, LocalBeadCoordinates
from .farfield_data import FarfieldData


@njit(cache=True)
def _exec_lookup(source, lookup, r, c):
    """
    Compile the retrieval of the lookup table with Numba, as it's slightly
    faster
    """
    return source[:, lookup[r, c]]


class AssociatedLegendreData:
    __slots__ = (
        "_associated_legendre",
        "_associated_legendre_over_sin_theta",
        "_associated_legendre_dtheta",
        "_inverse",
    )

    def __init__(
        self,
        associated_legendre: np.ndarray,
        associated_legendre_over_sin_theta: np.ndarray,
        associated_legendre_dtheta: np.ndarray,
        inverse: np.ndarray,
    ) -> None:
        self._associated_legendre = associated_legendre
        self._associated_legendre_over_sin_theta = associated_legendre_over_sin_theta
        self._associated_legendre_dtheta = associated_legendre_dtheta
        self._inverse = inverse

    def associated_legendre(self, r: int, c: int):
        return _exec_lookup(self._associated_legendre, self._inverse, r, c)

    def associated_legendre_over_sin_theta(self, r: int, c: int):
        return _exec_lookup(self._associated_legendre_over_sin_theta, self._inverse, r, c)

    def associated_legendre_dtheta(self, r: int, c: int):
        return _exec_lookup(self._associated_legendre_dtheta, self._inverse, r, c)


@njit(cache=True, parallel=True)
def _loop_over_rotations(
    local_coords_stacked: np.ndarray,
    radii: np.ndarray,
    aperture: np.ndarray,
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
):
    """
    Find all possible values of cos(theta), where theta is the angle of a coordinate with the z
    axis. Do this by looping over all possible rotations of the local coordinate system, as dictated
    by the plane waves in the focus, and calculating cos(theta) for every coordinate and every
    rotation.
    """
    rows, cols = np.nonzero(aperture)

    # Weird construct to work around tuple unpacking bug in Numba
    # https://github.com/numba/numba/issues/8772
    shape = (aperture.shape[0], aperture.shape[1], radii.size)
    local_cos_theta = np.zeros(shape)

    index = radii == 0
    for r, c in zip(rows, cols):

        # Rotate the coordinate system such that the x-polarization on the
        # bead coincides with theta polarization in global space
        # however, cos(theta) is the same for phi polarization!
        # >>> A = (_R_th(cos_theta_bfp[p,m], sin_theta[p,m]) @
        # >>>     _R_phi(cos_phi[p,m], -sin_phi[p,m]))
        # >>> coords = A @ local_coords_stacked
        # >>> z = coords[2, :]

        # The following line is equal to z = coords[2, :] after doing the transform with A
        z = (
            local_coords_stacked[2, :] * cos_theta[r, c]
            - (
                local_coords_stacked[0, :] * cos_phi[r, c]
                + local_coords_stacked[1, :] * sin_phi[r, c]
            )
            * sin_theta[r, c]
        )

        # Retrieve an array of all values of cos(theta)
        local_cos_theta[r, c, :] = z / radii  # cos(theta)
        local_cos_theta[r, c, index] = 1

    return local_cos_theta


@njit(cache=True, parallel=True)
def _do_legendre_calc(
    unique_cos_theta: np.ndarray,
    unique_abs_cos_theta: np.ndarray,
    sign_inv: np.ndarray,
    max_negative_index: int,
    n_orders: int,
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
    parity = (-1) ** np.arange(n_orders)
    sin_theta = (((1 + unique_abs_cos_theta) * (1 - unique_abs_cos_theta)) ** 0.5)[sign_inv]
    for L in prange(1, n_orders + 1):
        alp_sin[L - 1, :] = associated_legendre_over_sin_theta(L, unique_abs_cos_theta)[sign_inv]
        alp[L - 1, :] = alp_sin[L - 1, :] * sin_theta

        if parity[L - 1] == -1 and max_negative_index is not None:
            alp[L - 1, : max_negative_index + 1] *= -1
            alp_sin[L - 1, : max_negative_index + 1] *= -1

    for L in prange(1, n_orders + 1):
        alp_prev = alp_sin[L - 2, :] if L > 1 else None
        alp_deriv[L - 1, :] = associated_legendre_dtheta(
            L, unique_cos_theta, (alp_sin[L - 1, :], alp_prev)
        )

    return alp, alp_sin, alp_deriv


def calculate_legendre(
    location: CoordLocation,
    local_coordinates: LocalBeadCoordinates,
    farfield_data: FarfieldData,
    n_orders: int,
):
    """
    Find the value of cos(theta) for all local coordinates (x, y, z) after rotating
    the coordinates along all possible directions, defined by the angles in the
    back focal plane.
    """

    # Unpack data class members to a dict for Numba
    kwargs = {
        field.name: getattr(farfield_data, field.name)
        for field in fields(farfield_data)
        if field.name in ("aperture", "cos_theta", "sin_theta", "cos_phi", "sin_phi")
    }
    local_coords_stacked = local_coordinates.xyz_stacked(location)
    radii = local_coordinates.r(location)
    local_cos_theta = _loop_over_rotations(local_coords_stacked, radii, **kwargs)

    shape = local_cos_theta.shape
    local_cos_theta = np.reshape(local_cos_theta, local_cos_theta.size)
    # rounding errors may make cos(theta) > 1 or < -1. Fix it to [-1..1]
    local_cos_theta[local_cos_theta > 1] = 1
    local_cos_theta[local_cos_theta < -1] = -1

    # Get the unique values of cos(theta) in the array
    unique_cos_theta, inverse = np.unique(local_cos_theta, return_inverse=True)
    inverse = np.reshape(inverse, shape)

    # Get the signs of the cosines, and use parity to reduce the computational
    # load
    max_negative_index = None
    try:
        max_negative_index = np.argmax(np.where(unique_cos_theta < 0)[0])
    except ValueError:
        pass

    unique_abs_cos_theta, sign_inv = np.unique(np.abs(unique_cos_theta), return_inverse=True)

    alp, alp_sin, alp_deriv = _do_legendre_calc(
        unique_cos_theta, unique_abs_cos_theta, sign_inv, max_negative_index, n_orders
    )
    return AssociatedLegendreData(alp, alp_sin, alp_deriv, inverse)

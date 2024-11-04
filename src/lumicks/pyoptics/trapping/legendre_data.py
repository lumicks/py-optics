from dataclasses import fields

import numpy as np
from numba import njit, prange

from ..farfield_data import FarfieldData
from ..mathutils.associated_legendre import (
    associated_legendre_dtheta,
    associated_legendre_over_sin_theta,
)
from .local_coordinates import Coordinates


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
def _alp_sin_theta_with_parity(
    unique_abs_cos_theta: np.ndarray,
    sign_inv: np.ndarray,
    max_negative_index: int,
    n_orders: int,
):
    """
    Numba compatible function that computes Associated Legendre Polynomials for
    n_order orders, plus derived quantities
    """

    # Allocate memory for the Associated Legendre Polynomials divided by sin(theta)
    alp_sin_theta = np.zeros((n_orders, sign_inv.size))

    # Parity for order L: parity[0] == 1, for L == 1
    parity = (-1) ** np.arange(n_orders)
    for L in prange(1, n_orders + 1):
        alp_sin_theta[L - 1, :] = associated_legendre_over_sin_theta(L, unique_abs_cos_theta)[
            sign_inv
        ]

        if parity[L - 1] == -1 and max_negative_index is not None:
            alp_sin_theta[L - 1, : max_negative_index + 1] *= -1

    return alp_sin_theta


def calculate_legendre(
    coordinates: Coordinates,
    farfield_data: FarfieldData,
    n_orders: int,
):
    """
    Calculate the value of the Associated Legendre Functions of order `n_orders` and degree 1 for
    the unique values of cos(theta). These values are found by rotating all local coordinates (x, y,
    z) over all possible angles, as defined by the plane waves coming from the back focal plane.
    """

    # Unpack data class members to a dict for Numba
    farfield_args = {
        field.name: getattr(farfield_data, field.name)
        for field in fields(farfield_data)
        if field.name in ("aperture", "cos_theta", "sin_theta", "cos_phi", "sin_phi")
    }
    local_cos_theta = _loop_over_rotations(coordinates.xyz_stacked, coordinates.r, **farfield_args)

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

    alp_sin_theta = _alp_sin_theta_with_parity(
        unique_abs_cos_theta, sign_inv, max_negative_index, n_orders
    )
    alp_dtheta = np.empty_like(alp_sin_theta)
    associated_legendre_dtheta(unique_cos_theta, alp_sin_theta, alp_dtheta)
    return (alp_sin_theta, inverse), (alp_dtheta, inverse)

from dataclasses import dataclass

import numpy as np
from ..mathutils.vector import outer_product


@dataclass
class NearfieldData:
    """A data class to store the near field and everything else that is required to compute the far
    field by integration over the sampling (integration) points.

    Parameters
    ----------

    x : np.ndarray
        The x part of the sampling (integration) locations `x`, `y` and `z`
    y : np.ndarray
        Same as `x`, but y component.
    z : np.ndarray
        Same as `y`, but z component.
    normals: np.ndarray
        The outward normal at every location (x, y, z). First index is x, second index is y, last
        index is z.
    weight : np.ndarray
        The integration weight of every point. Used to calculate the integral by a (weighted)
        summation of points.
    E : np.ndarray
        The electric field at each position, as E[axis, position] and axis = [0, 1, 2] corresponding
        to x, y & z.
    H : np.ndarray
        Like E, but for the magnetic field.
    k : float
        The wave number inside the medium, equal to 2 * pi * n_medium / lamnda_vac
    eta : float
        Impedance of space inside the medium: (mu_0 / epsilon_0)**0.5 / n_medium

    """

    x: np.ndarray  # x-locations of integration points
    y: np.ndarray  # y-locations of integration points
    z: np.ndarray  # z-locations of integration points
    normals: np.ndarray  # Normals at integration points
    weight: np.ndarray  # weights of integration points
    E: np.ndarray  # Ex, Ey, Ez-fields at integration points
    H: np.ndarray  # Hx, Hy, Hz-fields at integration points
    k: float
    eta: float


def near_field_to_far_field(
    near_field_data: NearfieldData, cos_theta, sin_theta, cos_phi, sin_phi, r, aperture
):
    xb, yb, zb, k = [getattr(near_field_data, item) for item in "xyzk"]
    w = near_field_data.weight * 4 * np.pi
    eta = near_field_data.eta
    J_x, J_y, J_z = outer_product(near_field_data.normals, near_field_data.H)
    M_x, M_y, M_z = outer_product(near_field_data.E, near_field_data.normals)
    J_x, J_y, J_z, M_x, M_y, M_z = [np.atleast_1d(arr) for arr in (J_x, J_y, J_z, M_x, M_y, M_z)]
    E_theta, E_phi = _equivalent_currents_to_farfield(
        [xb, yb, zb],
        [J_x, J_y, J_z],
        [M_x, M_y, M_z],
        w,
        [cos_theta, sin_theta, cos_phi, sin_phi],
        r,
        aperture,
        k,
        eta,
    )

    return E_theta, E_phi


def _equivalent_currents_to_farfield(locations, J, M, weights, cosines, r, aperture, k, eta):
    cos_theta, sin_theta, cos_phi, sin_phi = cosines
    xb, yb, zb = locations
    Jx, Jy, Jz = J
    Mx, My, Mz = M
    L_phi, L_theta, N_phi, N_theta = [
        np.zeros_like(cos_theta, dtype="complex128") for _ in range(4)
    ]
    for idx in np.flatnonzero(aperture):
        weighted_phasor = weights * np.exp(
            (1j * k)
            * (
                (xb * cos_phi.flat[idx] + yb * sin_phi.flat[idx]) * sin_theta.flat[idx]
                + zb * cos_theta.flat[idx]
            )
        )
        L_phi.flat[idx] = (
            (-Mx * sin_phi.flat[idx] + My * cos_phi.flat[idx]) * weighted_phasor
        ).sum()
        L_theta.flat[idx] = (
            (
                (Mx * cos_phi.flat[idx] + My * sin_phi.flat[idx]) * cos_theta.flat[idx]
                - Mz * sin_theta.flat[idx]
            )
            * weighted_phasor
        ).sum()
        N_phi.flat[idx] = (
            (-Jx * sin_phi.flat[idx] + Jy * cos_phi.flat[idx]) * weighted_phasor
        ).sum()
        N_theta.flat[idx] = (
            (
                (Jx * cos_phi.flat[idx] + Jy * sin_phi.flat[idx]) * cos_theta.flat[idx]
                - Jz * sin_theta.flat[idx]
            )
            * weighted_phasor
        ).sum()
    G = (1j * k * np.exp(1j * k * r)) / (4 * np.pi * r)
    E_theta = G * (L_phi + eta * N_theta)
    E_phi = -G * (L_theta - eta * N_phi)
    return E_theta, E_phi

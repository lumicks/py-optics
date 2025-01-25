from typing import Optional, Tuple

import numpy as np
from scipy.constants import epsilon_0, mu_0
import matplotlib.pyplot as plt

from ..mathutils.integration import (
    determine_integration_order,
    get_integration_locations,
    get_nearest_order,
)
from ..objective import Objective
from ..trapping.bead import Bead
from ..trapping.interface import focus_field_factory
from ..trapping.local_coordinates import LocalBeadCoordinates
from ..field_distributions.dipole import field_dipole_y, field_dipole_z, field_dipole_x


def outer_product(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x, y, z = x1
    vx, vy, vz = x2

    px = y * vz - z * vy
    py = z * vx - x * vz
    pz = x * vy - y * vx
    return [px, py, pz]


def farfield_factory(
    f_input_field,
    objective: Objective,
    condenser: Objective,
    bead: Bead,
    objective_bfp_sampling_n: int = 31,
    condenser_bfp_sampling_n: int = 150,
    num_orders: Optional[int] = None,
    integration_order: Optional[int] = None,
    method: str = "lebedev-laikov",
):
    if bead.n_medium != objective.n_medium:
        raise ValueError("The immersion medium of the bead and the objective have to be the same")

    n_orders = bead.number_of_orders if num_orders is None else max(int(num_orders), 1)

    if integration_order is not None:
        # Use user's integration order
        integration_order = np.max((2, int(integration_order)))
        if method == "lebedev-laikov":
            # Find nearest integration order that is equal or greater than the user's
            integration_order = get_nearest_order(integration_order)
    else:
        integration_order = determine_integration_order(method, n_orders)
    x, y, z, w = get_integration_locations(integration_order, method)
    xb, yb, zb = [c * bead.bead_diameter * 0.51 for c in (x, y, z)]

    local_coordinates = LocalBeadCoordinates(
        xb, yb, zb, bead.bead_diameter, (0.0, 0.0, 0.0), grid=False
    )
    external_fields_func = focus_field_factory(
        objective,
        bead,
        n_orders,
        objective_bfp_sampling_n,
        f_input_field,
        local_coordinates,
        False,
    )

    eta = (mu_0 / epsilon_0) ** 0.5 / bead.n_medium

    bfp = condenser.get_back_focal_plane_coordinates(condenser_bfp_sampling_n)
    cos_theta, sin_theta, cos_phi, sin_phi, aperture = condenser.get_farfield_cosines(bfp)
    # fig, ax = plt.subplots(1, 5)
    # for idx, field in enumerate([cos_theta, sin_theta, cos_phi, sin_phi, aperture]):
    #     p = ax[idx].imshow(field)
    #     fig.colorbar(p)
    # fig.show()

    k = 2 * np.pi * bead.n_medium / bead.lambda_vac

    def farfield(bead_center: Tuple[float, float, float], num_threads: Optional[int] = None):
        bead_center = np.atleast_2d(bead_center)
        # shape = (numbers of bead positions, number of field coordinates)
        Ex, Ey, Ez, Hx, Hy, Hz = field_dipole_z(1, bead.n_medium, bead.lambda_vac, xb, yb, zb)
        # external_fields_func(bead_center, True, True, True, num_threads)
        J_x, J_y, J_z = outer_product([x, y, z], [Hx, Hy, Hz])
        M_x, M_y, M_z = outer_product([Ex, Ey, Ez], [x, y, z])
        L_phi, L_theta, N_phi, N_theta = [
            np.zeros_like(cos_theta, dtype="complex128") for _ in range(4)
        ]
        E_theta, E_phi = [], []
        for bead_idx, (bead_pos_x, bead_pos_y, bead_pos_z) in enumerate(bead_center):
            for idx in np.flatnonzero(aperture):
                weighted_phasor = w * np.exp(
                    (1j * k)
                    * (
                        (
                            (xb + bead_pos_x) * cos_phi.flat[idx]
                            + (yb + bead_pos_y) * sin_phi.flat[idx]
                        )
                        * sin_theta.flat[idx]
                        + (zb + bead_pos_z) * cos_theta.flat[idx]
                    )
                )
                L_phi.flat[idx] = (
                    (-M_x[bead_idx] * sin_phi.flat[idx] + M_y[bead_idx] * cos_phi.flat[idx])
                    * weighted_phasor
                ).sum()
                L_theta.flat[idx] = (
                    (
                        (M_x[bead_idx] * cos_phi.flat[idx] + M_y[bead_idx] * sin_phi.flat[idx])
                        * cos_theta.flat[idx]
                        - M_z[bead_idx] * sin_theta.flat[idx]
                    )
                    * weighted_phasor
                ).sum()
                N_phi.flat[idx] = (
                    (-J_x[bead_idx] * sin_phi.flat[idx] + J_y[bead_idx] * cos_phi.flat[idx])
                    * weighted_phasor
                ).sum()
                N_theta.flat[idx] = (
                    (
                        (J_x[bead_idx] * cos_phi.flat[idx] + J_y[bead_idx] * sin_phi.flat[idx])
                        * cos_theta.flat[idx]
                        - J_z[bead_idx] * sin_theta.flat[idx]
                    )
                    * weighted_phasor
                ).sum()

            fig, ax = plt.subplots(1, 4)
            for idx, field in enumerate([L_phi, L_theta, N_phi, N_theta]):
                fig.colorbar(ax[idx].imshow(np.abs(field)), ax=ax[idx])
            fig.show()
            E_theta.append(
                (1j * k * np.exp(1j * k * condenser.focal_length))
                / (4 * np.pi * condenser.focal_length)
                * (4 * np.pi)
                * (L_phi + eta * N_theta)
            )
            E_phi.append(
                (-1j * k * np.exp(1j * k * condenser.focal_length))
                / (4 * np.pi * condenser.focal_length)
                * (4 * np.pi)
                * (L_theta - eta * N_phi)
            )

        return np.squeeze(np.asarray(E_theta)), np.squeeze(np.asarray(E_phi))

    return farfield

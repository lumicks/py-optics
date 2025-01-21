import numpy as np
from ..objective import Objective
from ..trapping.bead import Bead
from ..trapping.interface import focus_field_factory


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
    _eps = EPS0 * bead.n_medium**2
    _mu = MU0

    n = np.vstack((x, y, z))
    bfp = condenser.get_back_focal_plane_coordinates(condenser_bfp_sampling_n)
    cos_theta, sin_theta, cos_phi, sin_phi, aperture = condenser.get_farfield_cosines(bfp)
    k = 2 * np.pi * bead.n_medium / bead.lambda_vac

    def farfield(bead_center: Tuple[float, float, float], num_threads: Optional[int] = None):
        bead_center = np.atleast_2d(bead_center)
        # shape = (numbers of bead positions, number of field coordinates)
        Ex, Ey, Ez, Hx, Hy, Hz = external_fields_func(bead_center, True, True, True, num_threads)
        E = np.stack((Ex, Ey, Ez), axis=2)
        H = np.stack((Hx, Hy, Hz), axis=2)
        Ms, Js = [field - np.sum(field * n.reshape(bead_center.shape[0], x.size, 3), axis=2) * field for field in (E, H)]
        Ms *= -1.0
        L_phi, L_theta, N_phi, N_theta = [
            np.zeros_like(cos_theta, dtype="complex128") for _ in range(4)
        ]
        for idx in range(len(cos_phi)):
            weighted_phasor = w * np.exp(
                (-1j * k)
                * (
                    ((xb + bead_center[0]) * cos_phi[idx] + (yb + bead_center[1]) * sin_phi[idx])
                    * sin_theta[idx]
                    + (zb + bead_center[2]) * cos_theta[idx]
                )
            )
            L_phi[idx] = ((-Ms[0] * sin_phi[idx] + Ms[1] * cos_phi[idx]) * weighted_phasor).sum()
            L_theta[idx] = (
                (
                    (Ms[0] * cos_phi[idx] + Ms[1] * sin_phi[idx]) * cos_theta[idx]
                    - Ms[2] * sin_theta[idx]
                )
                * weighted_phasor
            ).sum()
            N_phi[idx] = ((-Js[0] * sin_phi + Js[1] * cos_phi) * weighted_phasor).sum()
            N_theta[idx] = (
                ((Js[0] * cos_phi + Js[1] * sin_phi) * cos_theta - Js[2] * sin_theta)
                * weighted_phasor
            ).sum()

        eta = (_mu / _eps) ** 0.5
        E_theta = (1j * k * np.exp(1j * k * condenser.focal_length)) / (
            (4 * np.pi * condenser.focal_length) * (4 * np.pi) * (L_phi + eta * N_theta)
        )
        E_phi = (-1j * k * np.exp(1j * k * condenser.focal_length)) / (
            (4 * np.pi * condenser.focal_length) * (4 * np.pi) * (L_theta - eta * N_phi)
        )

    return farfield

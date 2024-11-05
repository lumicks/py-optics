from typing import Tuple

import numpy as np

from ..objective import Objective
from .bead import Bead
from .legendre_data import calculate_legendre
from .local_coordinates import (
    ExternalBeadCoordinates,
    InternalBeadCoordinates,
    LocalBeadCoordinates,
)
from .numba_implementation import do_loop
from .radial_data import calculate_external as calculate_external_radial_data
from .radial_data import calculate_internal as calculate_internal_radial_data


def focus_field_factory(
    objective: Objective,
    bead: Bead,
    n_orders: int,
    bfp_sampling_n: int,
    f_input_field: callable,
    local_coordinates: LocalBeadCoordinates,
    internal: bool,
):

    n_orders = n_orders
    bfp_sampling_n = bfp_sampling_n
    bfp_coords, bfp_fields = objective.sample_back_focal_plane(
        f_input_field=f_input_field, bfp_sampling_n=bfp_sampling_n
    )

    farfield_data = objective.back_focal_plane_to_farfield(bfp_coords, bfp_fields, bead.lambda_vac)

    local_coordinates = (
        InternalBeadCoordinates(local_coordinates)
        if internal
        else ExternalBeadCoordinates(local_coordinates)
    )
    r = local_coordinates.r
    radial_data = (
        calculate_internal_radial_data(bead.k1, r, n_orders)
        if internal
        else calculate_external_radial_data(bead.k, r, n_orders)
    )
    legendre_data, legendre_data_dtheta = calculate_legendre(
        local_coordinates, farfield_data, n_orders
    )
    an, bn = bead.ab_coeffs(n_orders)
    cn, dn = bead.cd_coeffs(n_orders)
    n_bead = bead.n_bead
    n_medium = bead.n_medium

    ks = bead.k * objective.NA / bead.n_medium
    dk = ks / (bfp_sampling_n - 1)
    phase_correction_factor = (-1j * objective.focal_length) * (
        np.exp(-1j * bead.k * objective.focal_length) * dk**2 / (2 * np.pi)
    )

    def calculate_field(
        bead_center: Tuple[float, float, float],
        calculate_electric_field: bool = True,
        calculate_magnetic_field: bool = False,
        calculate_total_field: bool = True,
    ):
        local_coords = local_coordinates.xyz_stacked
        region = np.reshape(local_coordinates.region, local_coordinates.coordinate_shape)

        # Always provide a (dummy) 2D numpy array to satisfy Numba (can't deal with None it seems),
        # but keep it small in case the parameter isn't used
        dummy = np.atleast_2d(0)

        # It's ugly but at least Numba compiles the loop
        E_field, H_field = do_loop(
            bead_center,
            an,
            bn,
            cn,
            dn,
            n_bead,
            n_medium,
            getattr(radial_data, "krH", dummy),
            getattr(radial_data, "dkrH_dkr", dummy),
            getattr(radial_data, "k0r", dummy),
            getattr(radial_data, "sphBessel", dummy),
            getattr(radial_data, "jn_over_k1r", dummy),
            getattr(radial_data, "jn_1", dummy),
            farfield_data.aperture,
            farfield_data.cos_theta,
            farfield_data.sin_theta,
            farfield_data.cos_phi,
            farfield_data.sin_phi,
            farfield_data.kx,
            farfield_data.ky,
            farfield_data.kz,
            farfield_data.Einf_theta,
            farfield_data.Einf_phi,
            legendre_data,
            legendre_data_dtheta,
            r,
            local_coords,
            internal,
            calculate_total_field,
            calculate_electric_field,
            calculate_magnetic_field,
        )

        E, H = [
            (
                [np.zeros(local_coordinates.coordinate_shape, dtype="complex128") for _ in range(3)]
                if calculate
                else None
            )
            for calculate in (calculate_electric_field, calculate_magnetic_field)
        ]
        for field, storage in zip((E, H), (E_field, H_field)):
            if field is not None:
                storage *= phase_correction_factor
                for idx, component in enumerate(field):
                    component[region] = storage[idx, :]

        ret_val = tuple()
        if calculate_electric_field:
            Ex, Ey, Ez = [np.squeeze(component) for component in E]
            ret_val = (Ex, Ey, Ez)

        if calculate_magnetic_field:
            Hx, Hy, Hz = [np.squeeze(component) for component in H]
            ret_val += (Hx, Hy, Hz)
        return ret_val

    return calculate_field

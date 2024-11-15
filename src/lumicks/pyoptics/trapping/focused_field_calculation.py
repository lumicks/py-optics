from dataclasses import fields
from typing import Optional, Tuple

import numpy as np
from numba.core.config import NUMBA_NUM_THREADS

from ..objective import Objective
from .bead import Bead
from .legendre_data import calculate_legendre
from .local_coordinates import (
    ExternalBeadCoordinates,
    InternalBeadCoordinates,
    LocalBeadCoordinates,
)
from .numba_implementation import external_coordinates_loop, internal_coordinates_loop
from .radial_data import calculate_external as calculate_external_radial_data
from .radial_data import calculate_internal as calculate_internal_radial_data
from .thread_limiter import thread_limiter


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
    farfield_as_dict = {
        f.name: getattr(farfield_data, f.name)
        for f in fields(farfield_data)
        if f.name not in ("kp")
    }
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
    radial_as_dict = {f.name: getattr(radial_data, f.name) for f in fields(radial_data)}

    legendre_data, legendre_data_dtheta = calculate_legendre(
        local_coordinates, farfield_data, n_orders
    )
    coeffs = bead.cd_coeffs(n_orders) if internal else bead.ab_coeffs(n_orders)
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
        num_threads: Optional[int] = None,
    ):
        local_coords = local_coordinates.xyz_stacked
        region = np.reshape(local_coordinates.region, local_coordinates.coordinate_shape)
        bead_center = np.atleast_2d(bead_center)
        if len(bead_center.shape) > 2:
            raise ValueError("Invalid argument for bead_center")
        num_threads = 1 if num_threads is None else min(max(1, int(num_threads)), NUMBA_NUM_THREADS)
        # Always provide a (dummy) 2D numpy array to satisfy Numba (can't deal with None it seems),
        # but keep it small in case the parameter isn't used
        with thread_limiter(num_threads):
            # It's ugly but at least Numba compiles the loop
            E_field, H_field = (
                internal_coordinates_loop(
                    bead_center,
                    coeffs,
                    n_bead,
                    **radial_as_dict,
                    **farfield_as_dict,
                    legendre_data=legendre_data,
                    legendre_data_dtheta=legendre_data_dtheta,
                    r=r,
                    local_coords=local_coords,
                    calculate_electric=calculate_electric_field,
                    calculate_magnetic=calculate_magnetic_field,
                    n_threads=num_threads,
                )
                if internal
                else external_coordinates_loop(
                    bead_center,
                    coeffs,
                    n_medium,
                    **radial_as_dict,
                    **farfield_as_dict,
                    legendre_data=legendre_data,
                    legendre_data_dtheta=legendre_data_dtheta,
                    r=r,
                    local_coords=local_coords,
                    total=calculate_total_field,
                    calculate_electric=calculate_electric_field,
                    calculate_magnetic=calculate_magnetic_field,
                    n_threads=num_threads,
                )
            )

        E, H = [
            (
                [
                    np.zeros(
                        (len(bead_center), *local_coordinates.coordinate_shape), dtype="complex128"
                    )
                    for _ in range(3)
                ]
                if calculate
                else None
            )
            for calculate in (calculate_electric_field, calculate_magnetic_field)
        ]
        for field, storage in zip((E, H), (E_field, H_field)):
            if field is not None:
                storage *= phase_correction_factor
                for pos_idx in range(len(bead_center)):
                    for idx, component in enumerate(field):
                        component[pos_idx, region] = storage[pos_idx, idx, :]

        ret_val = tuple()
        if calculate_electric_field:
            Ex, Ey, Ez = [np.squeeze(component) for component in E]
            ret_val = (Ex, Ey, Ez)

        if calculate_magnetic_field:
            Hx, Hy, Hz = [np.squeeze(component) for component in H]
            ret_val += (Hx, Hy, Hz)
        return ret_val

    return calculate_field

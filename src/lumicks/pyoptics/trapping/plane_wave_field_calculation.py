from dataclasses import fields
from typing import Tuple

import numpy as np

from ..farfield_data import FarfieldData
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


def _set_farfield(theta: float, phi: float, polarization: Tuple[float, float], k: float):
    """Create a FarFieldData object that contains a single pixel (= single
    plane wave) with angles `theta` and `phi`, and with amplitude and polarization (E_theta, E_phi)
    given by `polarization`"""
    cos_theta = np.atleast_2d(np.cos(theta))
    sin_theta = np.atleast_2d(np.sin(theta))
    cos_phi = np.atleast_2d(np.cos(phi))
    sin_phi = np.atleast_2d(np.sin(phi))
    kz = k * cos_theta
    kp = k * sin_theta
    ky = -kp * sin_phi
    kx = -kp * cos_phi

    return FarfieldData(
        Einf_theta=np.atleast_2d(polarization[0]) * kz,
        Einf_phi=np.atleast_2d(polarization[1]) * kz,
        weights=np.atleast_2d(1.0),
        cos_theta=cos_theta,
        sin_theta=sin_theta,
        cos_phi=cos_phi,
        sin_phi=sin_phi,
        kz=kz,
        ky=ky,
        kx=kx,
        kp=kp,
    )


def plane_wave_field_factory(
    bead: Bead,
    n_orders: int,
    theta: float,
    phi: float,
    local_coordinates: LocalBeadCoordinates,
    internal: bool,
):
    local_coordinates = (
        InternalBeadCoordinates(local_coordinates)
        if internal
        else ExternalBeadCoordinates(local_coordinates)
    )
    farfield_data = _set_farfield(theta=theta, phi=phi, polarization=[0, 0], k=bead.k)
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

    def calculate_field(
        polarization: Tuple[float, float],
        calculate_electric_field: bool = True,
        calculate_magnetic_field: bool = False,
        calculate_total_field: bool = True,
    ):
        farfield_data = _set_farfield(theta=theta, phi=phi, polarization=polarization, k=bead.k)
        farfield_as_dict = {
            f.name: getattr(farfield_data, f.name)
            for f in fields(farfield_data)
            if f.name not in ("kp")
        }
        local_coords = local_coordinates.xyz_stacked
        region = np.reshape(local_coordinates.region, local_coordinates.coordinate_shape)

        # Since we're not stacking plane waves, there's no need for multi-threading
        n_threads = 1
        with thread_limiter(n_threads):
            # The function `do_loop` doesn't actually loop over anything here, as it's just a single
            # plane wave. But we re-use the code that can assemble the plane-wave response for a set of
            # plane waves from any angle to assemble the field for a single one.
            E_field, H_field = (
                internal_coordinates_loop(
                    np.atleast_2d((0.0, 0.0, 0.0)),
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
                    n_threads=n_threads,
                )
                if internal
                else external_coordinates_loop(
                    np.atleast_2d((0.0, 0.0, 0.0)),
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
                    n_threads=n_threads,
                )
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
                for idx, component in enumerate(field):
                    component[region] = storage[0, idx, :]

        ret_val = tuple()
        if calculate_electric_field:
            Ex, Ey, Ez = [np.squeeze(component) for component in E]
            ret_val = (Ex, Ey, Ez)

        if calculate_magnetic_field:
            Hx, Hy, Hz = [np.squeeze(component) for component in H]
            ret_val += (Hx, Hy, Hz)
        return ret_val

    return calculate_field

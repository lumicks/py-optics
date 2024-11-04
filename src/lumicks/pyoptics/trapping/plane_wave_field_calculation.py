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
from .numba_loop import _do_loop
from .radial_data import calculate_external as calculate_external_radial_data
from .radial_data import calculate_internal as calculate_internal_radial_data


def _set_farfield(theta: float, phi: float, polarization: Tuple[float, float], k: float):
    # Create a FarFieldData object that contains a single pixel == single
    # plane wave with angles theta and phi and with amplitude and
    # polarization (E_theta, E_phi) given by `polarization`
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
        aperture=np.atleast_2d(True),
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

    n_orders = n_orders

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
    legendre_data, legendre_data_dtheta = calculate_legendre(
        local_coordinates, farfield_data, n_orders
    )
    an, bn = bead.ab_coeffs(n_orders)
    cn, dn = bead.cd_coeffs(n_orders)
    n_bead = bead.n_bead
    n_medium = bead.n_medium

    def calculate_field(
        polarization: Tuple[float, float],
        calculate_electric_field: bool = True,
        calculate_magnetic_field: bool = False,
        calculate_total_field: bool = True,
    ):
        dummy = np.atleast_2d(0)
        farfield_data = _set_farfield(theta=theta, phi=phi, polarization=polarization, k=bead.k)

        local_coords = local_coordinates.xyz_stacked
        region = np.reshape(local_coordinates.region, local_coordinates.coordinate_shape)

        E_field, H_field = _do_loop(
            (0.0, 0.0, 0.0),
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

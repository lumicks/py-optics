import numpy as np
from numba import njit
from .implementation import (
    R_pol_R_th_R_phi,
    R_th_R_phi,
    internal_field_fixed_r,
    internal_H_field_fixed_r,
    scattered_field_fixed_r,
    scattered_H_field_fixed_r,
)


@njit(cache=True)
def _do_loop(
    bead_center,
    an,
    bn,
    cn,
    dn,
    n_bead,
    n_medium,
    krH,
    dkrH_dkr,
    k0r,
    sphBessel,
    jn_over_k1r,
    jn_1,
    aperture,
    cos_theta,
    sin_theta,
    cos_phi,
    sin_phi,
    kx,
    ky,
    kz,
    Einf_theta,
    Einf_phi,
    legendre_data,
    legendre_data_dtheta,
    r,
    local_coords,
    internal: bool,
    total: bool,
    calculate_electric: bool,
    calculate_magnetic: bool,
):
    local_cos_theta = np.empty(r.shape)
    plane_wave_response_xyz = np.empty((3, r.size), dtype="complex128")
    # Mask r == 0:
    r_eq_zero = r == 0

    dummy = np.zeros((1, 1), dtype="complex128")
    field_storage_E = np.zeros_like(plane_wave_response_xyz) if calculate_electric else dummy
    field_storage_H = np.zeros_like(plane_wave_response_xyz) if calculate_magnetic else dummy

    # Skip points outside aperture
    rows, cols = np.nonzero(aperture)
    if r.size > 0:
        for row, col in zip(rows, cols):
            matrices = [
                R_th_R_phi(
                    cos_theta[row, col],
                    sin_theta[row, col],
                    cos_phi[row, col],
                    -sin_phi[row, col],
                ),
                R_pol_R_th_R_phi(
                    cos_theta[row, col],
                    sin_theta[row, col],
                    cos_phi[row, col],
                    -sin_phi[row, col],
                ),
            ]

            E0 = [Einf_theta[row, col], Einf_phi[row, col]]

            for polarization in range(2):
                A = matrices[polarization]
                coords = A @ local_coords
                x = coords[0, :]
                y = coords[1, :]
                z = coords[2, :]
                if polarization == 0:
                    local_cos_theta[:] = z / r
                    local_cos_theta[r_eq_zero] = 1
                    np.clip(local_cos_theta, a_max=1, a_min=-1, out=local_cos_theta)

                    # Expand the legendre derivatives from the unique version
                    # of cos(theta)
                    local_sin_theta = ((1 + local_cos_theta) * (1 - local_cos_theta)) ** 0.5
                    indices = legendre_data[1][row, col]
                    alp_sin_expanded = legendre_data[0][:, indices]
                    alp_expanded = alp_sin_expanded * local_sin_theta
                    indices = legendre_data_dtheta[1][row, col]
                    alp_deriv_expanded = legendre_data_dtheta[0][:, indices]

                rho_l = np.hypot(x, y)
                cosP = x / rho_l
                sinP = y / rho_l
                where = rho_l == 0
                cosP[where] = 1
                sinP[where] = 0

                if calculate_electric:
                    if internal:
                        plane_wave_response = internal_field_fixed_r(
                            cn,
                            dn,
                            sphBessel,
                            jn_over_k1r,
                            jn_1,
                            alp_expanded,
                            alp_sin_expanded,
                            alp_deriv_expanded,
                            local_cos_theta,
                            local_sin_theta,
                            cosP,
                            sinP,
                        )
                    else:
                        plane_wave_response = scattered_field_fixed_r(
                            an,
                            bn,
                            krH,
                            dkrH_dkr,
                            k0r,
                            alp_expanded,
                            alp_sin_expanded,
                            alp_deriv_expanded,
                            local_cos_theta,
                            local_sin_theta,
                            cosP,
                            sinP,
                            total,
                        )

                    plane_wave_response_xyz[:] = A.T.astype("complex128") @ plane_wave_response
                    plane_wave_response_xyz *= (
                        E0[polarization]
                        * np.exp(
                            1j
                            * (
                                kx[row, col] * bead_center[0]
                                + ky[row, col] * bead_center[1]
                                + kz[row, col] * bead_center[2]
                            )
                        )
                        / kz[row, col]
                    )

                    field_storage_E += plane_wave_response_xyz

                if calculate_magnetic:
                    if internal:
                        plane_wave_response = internal_H_field_fixed_r(
                            cn,
                            dn,
                            sphBessel,
                            jn_over_k1r,
                            jn_1,
                            alp_expanded,
                            alp_sin_expanded,
                            alp_deriv_expanded,
                            local_cos_theta,
                            local_sin_theta,
                            cosP,
                            sinP,
                            n_bead,
                        )
                    else:
                        plane_wave_response = scattered_H_field_fixed_r(
                            an,
                            bn,
                            krH,
                            dkrH_dkr,
                            k0r,
                            alp_expanded,
                            alp_sin_expanded,
                            alp_deriv_expanded,
                            local_cos_theta,
                            local_sin_theta,
                            cosP,
                            sinP,
                            n_medium,
                            total,
                        )
                    plane_wave_response_xyz[:] = A.T.astype("complex128") @ plane_wave_response
                    plane_wave_response_xyz *= (
                        E0[polarization]
                        * np.exp(
                            1j
                            * (
                                kx[row, col] * bead_center[0]
                                + ky[row, col] * bead_center[1]
                                + kz[row, col] * bead_center[2]
                            )
                        )
                        / kz[row, col]
                    )
                    # Accumulate the field for this plane wave and polarization
                    field_storage_H += plane_wave_response_xyz
    return field_storage_E, field_storage_H

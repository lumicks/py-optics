"""This module implements a numba-compatible (compilable) for loop that speeds up the summation of
plane waves, and the local fields as the subsequent response of each plane wave"""

import numpy as np
from numba import get_thread_id, njit, prange
from scipy.constants import mu_0 as MU0
from scipy.constants import speed_of_light as C


@njit(cache=True, parallel=True)
def external_coordinates_loop(
    bead_center,
    coeffs,
    n_medium,
    krH,
    dkrH_dkr,
    k0r,
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
    total: bool,
    calculate_electric: bool,
    calculate_magnetic: bool,
    n_threads: int,
):
    an, bn = coeffs
    n_orders = len(an)
    dummy = np.zeros((1, 1, 1, 1), dtype="complex128")
    field_storage_E = (
        np.zeros((n_threads, len(bead_center), 3, r.size), dtype="complex128")
        if calculate_electric
        else dummy
    )
    field_storage_H = np.zeros_like(field_storage_E) if calculate_magnetic else dummy

    # Skip points outside aperture
    rows, cols = np.nonzero(aperture)
    if r.size > 0:
        for loop_idx in prange(rows.size):
            row, col = rows[loop_idx], cols[loop_idx]
            t_id = get_thread_id()
            matrices = [
                _R_th_R_phi(
                    cos_theta[row, col],
                    sin_theta[row, col],
                    cos_phi[row, col],
                    -sin_phi[row, col],
                ),
                _R_pol_R_th_R_phi(
                    cos_theta[row, col],
                    sin_theta[row, col],
                    cos_phi[row, col],
                    -sin_phi[row, col],
                ),
            ]
            local_cos_theta = np.empty(r.size)
            local_sin_theta = np.empty_like(local_cos_theta)
            alp_sin_expanded = np.empty((n_orders, r.size))
            alp_expanded = np.empty_like(alp_sin_expanded)
            alp_deriv_expanded = np.empty_like(alp_sin_expanded)

            E0 = [Einf_theta[row, col], Einf_phi[row, col]]

            for polarization in range(2):
                A = matrices[polarization]
                coords = A @ local_coords
                x = coords[0, :]
                y = coords[1, :]
                z = coords[2, :]
                if polarization == 0:
                    local_cos_theta[:] = z / r
                    np.clip(local_cos_theta, a_max=1, a_min=-1, out=local_cos_theta)

                    # Expand the legendre derivatives from the unique version of cos(theta)
                    local_sin_theta[:] = ((1 + local_cos_theta) * (1 - local_cos_theta)) ** 0.5
                    indices = legendre_data[1][row, col]
                    alp_sin_expanded[:] = legendre_data[0][:, indices]
                    alp_expanded[:] = alp_sin_expanded * local_sin_theta
                    indices = legendre_data_dtheta[1][row, col]
                    alp_deriv_expanded[:] = legendre_data_dtheta[0][:, indices]

                rho_l = np.hypot(x, y)
                cosP = x / rho_l
                sinP = y / rho_l
                where = rho_l == 0
                cosP[where] = 1
                sinP[where] = 0

                if calculate_electric:
                    plane_wave_response = _scattered_electric_field(
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

                    plane_wave_response_xyz = A.T.astype("complex128") @ plane_wave_response

                    for idx in range(len(bead_center)):
                        field_storage_E[t_id, idx] += plane_wave_response_xyz * (
                            E0[polarization]
                            * np.exp(
                                1j
                                * (
                                    kx[row, col] * bead_center[idx][0]
                                    + ky[row, col] * bead_center[idx][1]
                                    + kz[row, col] * bead_center[idx][2]
                                )
                            )
                            / kz[row, col]
                        )

                if calculate_magnetic:
                    plane_wave_response = _scattered_magnetic_field(
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
                    plane_wave_response_xyz = A.T.astype("complex128") @ plane_wave_response
                    for idx in range(len(bead_center)):
                        field_storage_H[t_id, idx] += plane_wave_response_xyz * (
                            E0[polarization]
                            * np.exp(
                                1j
                                * (
                                    kx[row, col] * bead_center[idx][0]
                                    + ky[row, col] * bead_center[idx][1]
                                    + kz[row, col] * bead_center[idx][2]
                                )
                            )
                            / kz[row, col]
                        )

    return np.sum(field_storage_E, axis=0), np.sum(field_storage_H, axis=0)


@njit(cache=True, parallel=True)
def internal_coordinates_loop(
    bead_center,
    coeffs,
    n_bead,
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
    calculate_electric: bool,
    calculate_magnetic: bool,
    n_threads: int,
):
    cn, dn = coeffs
    n_orders = len(cn)
    # plane_wave_response_xyz = np.empty((3, r.size), dtype="complex128")
    # Mask r == 0:
    r_eq_zero = r == 0
    dummy = np.zeros((1, 1, 1, 1), dtype="complex128")
    field_storage_E = (
        np.zeros((n_threads, len(bead_center), 3, r.size), dtype="complex128")
        if calculate_electric
        else dummy
    )
    field_storage_H = np.zeros_like(field_storage_E) if calculate_magnetic else dummy

    # Skip points outside aperture
    rows, cols = np.nonzero(aperture)
    if r.size > 0:
        for loop_idx in prange(rows.size):
            row, col = rows[loop_idx], cols[loop_idx]
            t_id = get_thread_id()
            matrices = [
                _R_th_R_phi(
                    cos_theta[row, col],
                    sin_theta[row, col],
                    cos_phi[row, col],
                    -sin_phi[row, col],
                ),
                _R_pol_R_th_R_phi(
                    cos_theta[row, col],
                    sin_theta[row, col],
                    cos_phi[row, col],
                    -sin_phi[row, col],
                ),
            ]
            local_cos_theta = np.empty(r.size)
            local_sin_theta = np.empty_like(local_cos_theta)
            alp_sin_expanded = np.empty((n_orders, r.size))
            alp_expanded = np.empty_like(alp_sin_expanded)
            alp_deriv_expanded = np.empty_like(alp_sin_expanded)

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

                    # Expand the legendre derivatives from the unique version of cos(theta)
                    local_sin_theta[:] = ((1 + local_cos_theta) * (1 - local_cos_theta)) ** 0.5
                    indices = legendre_data[1][row, col]
                    alp_sin_expanded[:] = legendre_data[0][:, indices]
                    alp_expanded[:] = alp_sin_expanded * local_sin_theta
                    indices = legendre_data_dtheta[1][row, col]
                    alp_deriv_expanded[:] = legendre_data_dtheta[0][:, indices]

                rho_l = np.hypot(x, y)
                cosP = x / rho_l
                sinP = y / rho_l
                where = rho_l == 0
                cosP[where] = 1
                sinP[where] = 0

                if calculate_electric:
                    plane_wave_response = _internal_electric_field(
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
                    plane_wave_response_xyz = A.T.astype("complex128") @ plane_wave_response

                    for idx in range(len(bead_center)):
                        field_storage_E[t_id, idx] += plane_wave_response_xyz * (
                            E0[polarization]
                            * np.exp(
                                1j
                                * (
                                    kx[row, col] * bead_center[idx][0]
                                    + ky[row, col] * bead_center[idx][1]
                                    + kz[row, col] * bead_center[idx][2]
                                )
                            )
                            / kz[row, col]
                        )

                if calculate_magnetic:
                    plane_wave_response = _internal_magnetic_field(
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
                    plane_wave_response_xyz = A.T.astype("complex128") @ plane_wave_response
                    for idx in range(len(bead_center)):
                        field_storage_H[t_id, idx] += plane_wave_response_xyz * (
                            E0[polarization]
                            * np.exp(
                                1j
                                * (
                                    kx[row, col] * bead_center[idx][0]
                                    + ky[row, col] * bead_center[idx][1]
                                    + kz[row, col] * bead_center[idx][2]
                                )
                            )
                            / kz[row, col]
                        )

    return np.sum(field_storage_E, axis=0), np.sum(field_storage_H, axis=0)


@njit(cache=True, parallel=False)
def _scattered_electric_field(
    an: np.ndarray,
    bn: np.ndarray,
    krh: np.ndarray,
    dkrh_dkr: np.ndarray,
    k0r: np.ndarray,
    alp: np.ndarray,
    alp_sin: np.ndarray,
    alp_deriv: np.ndarray,
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    total_field=True,
):
    """
    Calculate the scattered electric field for plane wave excitation, at the
    coordinates defined by r, theta and phi. Note that these are not explicitly
    used, but are implicitly present in the evaluations of the Hankel
    functions, Associated Legendre polynomials and derivatives.
    """
    # Radial, theta and phi-oriented fields
    Er = np.zeros((1, cos_theta.shape[0]), dtype="complex128")
    Et = np.zeros((1, cos_theta.shape[0]), dtype="complex128")
    Ep = np.zeros((1, cos_theta.shape[0]), dtype="complex128")

    L = np.arange(1, stop=an.size + 1)
    C1 = 1j ** (L + 1) * (2 * L + 1)
    C2 = C1 / (L * (L + 1))
    for L in range(an.size, 0, -1):
        Er += C1[L - 1] * an[L - 1] * krh[L - 1, :] * alp[L - 1, :]

        Et += C2[L - 1] * (
            an[L - 1] * dkrh_dkr[L - 1, :] * alp_deriv[L - 1, :]
            + 1j * bn[L - 1] * krh[L - 1, :] * alp_sin[L - 1, :]
        )

        Ep += C2[L - 1] * (
            an[L - 1] * dkrh_dkr[L - 1, :] * alp_sin[L - 1, :]
            + 1j * bn[L - 1] * krh[L - 1, :] * alp_deriv[L - 1, :]
        )

    Er *= -cos_phi / (k0r) ** 2
    Et *= -cos_phi / (k0r)
    Ep *= sin_phi / (k0r)
    # Cartesian components
    Ex = Er * sin_theta * cos_phi + Et * cos_theta * cos_phi - Ep * sin_phi
    Ey = Er * sin_theta * sin_phi + Et * cos_theta * sin_phi + Ep * cos_phi
    Ez = Er * cos_theta - Et * sin_theta
    if total_field:
        # Incident field (x-polarized)
        Ex += np.exp(1j * k0r * cos_theta)
    return np.concatenate((Ex, Ey, Ez), axis=0)


@njit(cache=True)
def _internal_electric_field(
    cn: np.ndarray,
    dn: np.ndarray,
    sphBessel: np.ndarray,
    jn_over_k1r: np.ndarray,
    jn_1: np.ndarray,
    alp: np.ndarray,
    alp_sin: np.ndarray,
    alp_deriv: np.ndarray,
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    cosP: np.ndarray,
    sinP: np.ndarray,
):
    """
    Calculate the internal electric field for plane wave excitation, at the
    coordinates defined by r, theta and phi. Note that these are not explicitly
    used, but are implicitly present in the evaluations of the Bessel
    functions, Associated Legendre polynomials and derivatives.
    """
    # Radial, theta and phi-oriented fields
    Er = np.zeros((1, cos_theta.shape[0]), dtype="complex128")
    Et = np.zeros((1, cos_theta.shape[0]), dtype="complex128")
    Ep = np.zeros((1, cos_theta.shape[0]), dtype="complex128")

    for n in range(cn.size, 0, -1):
        Er += -(1j ** (n + 1) * (2 * n + 1) * alp[n - 1, :] * dn[n - 1] * jn_over_k1r[n - 1, :])

        Et += (1j**n * (2 * n + 1) / (n * (n + 1))) * (
            cn[n - 1] * alp_sin[n - 1, :] * sphBessel[n - 1, :]
            - 1j * dn[n - 1] * alp_deriv[n - 1, :] * (jn_1[n - 1, :] - n * jn_over_k1r[n - 1, :])
        )

        Ep += (-(1j**n) * (2 * n + 1) / (n * (n + 1))) * (
            cn[n - 1] * alp_deriv[n - 1, :] * sphBessel[n - 1, :]
            - 1j * dn[n - 1] * alp_sin[n - 1, :] * (jn_1[n - 1, :] - n * jn_over_k1r[n - 1, :])
        )

    Er *= -cosP
    Et *= -cosP
    Ep *= -sinP
    # Cartesian components
    Ex = Er * sin_theta * cosP + Et * cos_theta * cosP - Ep * sinP
    Ey = Er * sin_theta * sinP + Et * cos_theta * sinP + Ep * cosP
    Ez = Er * cos_theta - Et * sin_theta
    return np.concatenate((Ex, Ey, Ez), axis=0)


@njit(cache=True)
def _scattered_magnetic_field(
    an: np.ndarray,
    bn: np.ndarray,
    krh: np.ndarray,
    dkrh_dkr: np.ndarray,
    k0r: np.ndarray,
    alp: np.ndarray,
    alp_sin: np.ndarray,
    alp_deriv: np.ndarray,
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    cosP: np.ndarray,
    sinP: np.ndarray,
    n_medium: float,
    total_field=True,
):
    """
    Calculate the scattered magnetic field for plane wave excitation, at the
    coordinates defined by r, theta and phi. Note that these are not explicitly
    used, but are implicitly present in the evaluations of the Hankel
    functions, Associated Legendre polynomials and derivatives.
    """
    # Radial, theta and phi-oriented fields
    Hr = np.zeros((1, cos_theta.shape[0]), dtype="complex128")
    Ht = np.zeros((1, cos_theta.shape[0]), dtype="complex128")
    Hp = np.zeros((1, cos_theta.shape[0]), dtype="complex128")

    L = np.arange(1, stop=an.size + 1)
    C1 = 1j**L * (2 * L + 1)
    C2 = C1 / (L * (L + 1))
    for L in range(an.size, 0, -1):
        Hr += C1[L - 1] * 1j * bn[L - 1] * krh[L - 1, :] * alp[L - 1, :]

        Ht += C2[L - 1] * (
            1j * bn[L - 1] * dkrh_dkr[L - 1, :] * alp_deriv[L - 1, :]
            - an[L - 1] * krh[L - 1, :] * alp_sin[L - 1, :]
        )

        Hp += C2[L - 1] * (
            1j * bn[L - 1] * dkrh_dkr[L - 1, :] * alp_sin[L - 1, :]
            - an[L - 1] * krh[L - 1, :] * alp_deriv[L - 1, :]
        )

    # Extra factor of -1 as B&H does not include the Condon–Shortley phase,
    # but our associated Legendre polynomials do include it
    Hr *= -sinP / (k0r) ** 2 * n_medium / (C * MU0)
    Ht *= -sinP / (k0r) * n_medium / (C * MU0)
    Hp *= -cosP / (k0r) * n_medium / (C * MU0)

    # Cartesian components
    Hx = Hr * sin_theta * cosP + Ht * cos_theta * cosP - Hp * sinP
    Hy = Hr * sin_theta * sinP + Ht * cos_theta * sinP + Hp * cosP
    Hz = Hr * cos_theta - Ht * sin_theta
    if total_field:
        # Incident field (E field x-polarized)
        Hy += np.exp(1j * k0r * cos_theta) * n_medium / (C * MU0)
    return np.concatenate((Hx, Hy, Hz), axis=0)


@njit(cache=True)
def _internal_magnetic_field(
    cn: np.ndarray,
    dn: np.ndarray,
    sphBessel: np.ndarray,
    jn_over_k1r: np.ndarray,
    jn_1: np.ndarray,
    alp: np.ndarray,
    alp_sin: np.ndarray,
    alp_deriv: np.ndarray,
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    cosP: np.ndarray,
    sinP: np.ndarray,
    n_bead: np.complex128,
):
    """
    Calculate the internal magnetic field for plane wave excitation, at the
    coordinates defined by r, theta and phi. Note that these are not explicitly
    used, but are implicitly present in the evaluations of the Bessel
    functions, Associated Legendre polynomials and derivatives.
    """
    # Radial, theta and phi-oriented fields
    Hr = np.zeros((1, cos_theta.shape[0]), dtype="complex128")
    Ht = np.zeros((1, cos_theta.shape[0]), dtype="complex128")
    Hp = np.zeros((1, cos_theta.shape[0]), dtype="complex128")

    for n in range(cn.size, 0, -1):
        Hr += 1j ** (n + 1) * (2 * n + 1) * alp[n - 1, :] * cn[n - 1] * jn_over_k1r[n - 1, :]

        Ht += (1j**n * (2 * n + 1) / (n * (n + 1))) * (
            dn[n - 1] * alp_sin[n - 1, :] * sphBessel[n - 1, :]
            - 1j * cn[n - 1] * alp_deriv[n - 1, :] * (jn_1[n - 1, :] - n * jn_over_k1r[n - 1, :])
        )

        Hp += (-(1j**n) * (2 * n + 1) / (n * (n + 1))) * (
            dn[n - 1] * alp_deriv[n - 1, :] * sphBessel[n - 1, :]
            - 1j * cn[n - 1] * alp_sin[n - 1, :] * (jn_1[n - 1, :] - n * jn_over_k1r[n - 1, :])
        )

    Hr *= sinP * n_bead / (C * MU0)
    Ht *= -sinP * n_bead / (C * MU0)
    Hp *= cosP * n_bead / (C * MU0)
    # Cartesian components
    Hx = Hr * sin_theta * cosP + Ht * cos_theta * cosP - Hp * sinP
    Hy = Hr * sin_theta * sinP + Ht * cos_theta * sinP + Hp * cosP
    Hz = Hr * cos_theta - Ht * sin_theta
    return np.concatenate((Hx, Hy, Hz), axis=0)


@njit
def _R_th_R_phi(cos_theta: float, sin_theta: float, cos_phi: float, sin_phi: float):
    """Creates a rotation matrix that first rotates over phi, then over theta. An explicit
    implementation is faster than an actual double matrix mulitplication

    Parameters
    ----------
    cos_theta : float
        Cosine of theta, with theta in radians
    sin_theta : float
        Sine of theta, with theta in radians
    cos_phi : float
        Cosine of phi, with phi in radians
    sin_phi : float
        Sine of phi, with phi in radians

    Returns
    -------
    np.ndarray[:, :]
        Two-dimensional Numpy array that implements R_theta @ R_phi
    """
    return np.asarray(
        [
            [cos_phi * cos_theta, -cos_theta * sin_phi, sin_theta],
            [sin_phi, cos_phi, 0],
            [-cos_phi * sin_theta, sin_phi * sin_theta, cos_theta],
        ]
    )


@njit
def _R_pol_R_th_R_phi(cos_theta: float, sin_theta: float, cos_phi: float, sin_phi: float):
    """Creates a rotation matrix that first rotates over phi, then over theta, then once more over
    (the then local) phi to flip the polarization. An explicit implementation is faster than an
    actual triple matrix mulitplication

    Parameters
    ----------
    ccos_theta : float
        Cosine of theta, with theta in radians
    sin_theta : float
        Sine of theta, with theta in radians
    cos_phi : float
        Cosine of phi, with phi in radians
    sin_phi : float
        Sine of phi, with phi in radians

    Returns
    -------
    np.ndarray[:, :]
        Two-dimensional Numpy array that implements R_pol @ R_theta @ R_phi
    """
    return np.asarray(
        [
            [sin_phi, cos_phi, 0],
            [-cos_phi * cos_theta, cos_theta * sin_phi, -sin_theta],
            [-cos_phi * sin_theta, sin_phi * sin_theta, cos_theta],
        ]
    )

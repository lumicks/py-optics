import numpy as np
from numba import njit
from scipy.constants import (
    mu_0 as _MU0,
    speed_of_light as _C
)

from .local_coordinates import LocalBeadCoordinates
from .radial_data import (
    ExternalRadialData,
    InternalRadialData
)
from .legendre_data import AssociatedLegendreData
from .objective import FarfieldData
from .bead import Bead


def calculate_fields(
    Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray,
    Hx: np.ndarray, Hy: np.ndarray, Hz: np.ndarray,
    bead: Bead = None,
    bead_center: tuple = (0, 0, 0),
    local_coordinates: LocalBeadCoordinates = None,
    farfield_data: FarfieldData = None,
    legendre_data: AssociatedLegendreData = None,
    external_radial_data: ExternalRadialData = None,
    internal_radial_data: InternalRadialData = None,
    internal: bool = False,
    total_field: bool = True,
    magnetic_field: bool = False,
):
    """Calculate the internal & external field from precalculated,
    but compressed, Legendre polynomials
    Compromise between speed and memory use"""
    if internal:
        r = local_coordinates.r_inside
        local_coords = local_coordinates.xyz_stacked(inside=True)
        region = np.atleast_1d(
            np.squeeze(local_coordinates.region_inside_bead)
        )
        n_orders = internal_radial_data.sphBessel.shape[0]
    else:
        r = local_coordinates.r_outside
        local_coords = local_coordinates.xyz_stacked(inside=False)
        region = np.atleast_1d(
            np.squeeze(local_coordinates.region_outside_bead)
        )
        n_orders = external_radial_data.krH.shape[0]
    if r.size == 0:
        return

    E = np.empty((3, local_coordinates.r.shape[1]), dtype='complex128')
    if magnetic_field:
        H = np.empty(E.shape, dtype='complex128')

    # preallocate memory for expanded legendre derivatives
    alp_expanded = np.empty((n_orders, r.size))
    alp_sin_expanded = np.empty(alp_expanded.shape)
    alp_deriv_expanded = np.empty(alp_expanded.shape)

    an, bn = bead.ab_coeffs(n_orders)
    cn, dn = bead.cd_coeffs(n_orders)

    n_pupil_samples = farfield_data.Einf_phi.shape[0]
    cosT = np.empty(r.shape)
    for m in range(n_pupil_samples):
        for p in range(n_pupil_samples):
            # Skip points outside aperture
            if not farfield_data.aperture[p, m]:
                continue
            matrices = [
                R_th(farfield_data.cos_theta[p, m],
                     farfield_data.sin_theta[p, m]) @
                R_phi(farfield_data.cos_phi[p, m],
                      -farfield_data.sin_phi[p, m]),
                R_phi(0, -1) @
                R_th(farfield_data.cos_theta[p, m],
                     farfield_data.sin_theta[p, m]) @
                R_phi(farfield_data.cos_phi[p, m],
                      -farfield_data.sin_phi[p, m])
            ]

            E0 = [farfield_data.Einf_theta[p, m], farfield_data.Einf_phi[p, m]]

            for polarization in range(2):
                A = matrices[polarization]
                coords = A @ local_coords
                x = coords[0, :]
                y = coords[1, :]
                z = coords[2, :]

                if polarization == 0:
                    if internal:
                        cosT[r > 0] = z[r > 0] / r[r > 0]
                        cosT[r == 0] = 1
                    else:
                        cosT[:] = z / r
                    cosT[cosT > 1] = 1
                    cosT[cosT < -1] = -1

                    # Expand the legendre derivatives from the unique version
                    # of cos(theta)
                    alp_expanded[:] = legendre_data.associated_legendre(p, m)
                    alp_sin_expanded[:] = \
                        legendre_data.associated_legendre_over_sin_theta(p, m)
                    alp_deriv_expanded[:] = \
                        legendre_data.associated_legendre_dtheta(p, m)

                rho_l = np.hypot(x, y)
                cosP = np.empty(rho_l.shape)
                sinP = np.empty(rho_l.shape)
                where = rho_l > 0
                cosP[where] = x[where] / rho_l[where]
                sinP[where] = y[where] / rho_l[where]
                cosP[rho_l == 0] = 1
                sinP[rho_l == 0] = 0

                E[:] = 0
                if internal:
                    E[:, region] = internal_field_fixed_r(
                        cn, dn,
                        internal_radial_data.sphBessel,
                        internal_radial_data.jn_over_k1r,
                        internal_radial_data.jn_1,
                        alp_expanded,
                        alp_sin_expanded, alp_deriv_expanded,
                        cosT, cosP, sinP
                    )
                else:
                    E[:, region] = scattered_field_fixed_r(
                        an, bn,
                        external_radial_data.krH,
                        external_radial_data.dkrH_dkr,
                        external_radial_data.k0r,
                        alp_expanded, alp_sin_expanded,
                        alp_deriv_expanded, cosT, cosP, sinP,
                        total_field
                    )

                E = np.matmul(A.T, E)
                E[:, region] *= E0[polarization] * np.exp(1j * (
                    farfield_data.kx[p, m] * bead_center[0] +
                    farfield_data.ky[p, m] * bead_center[1] +
                    farfield_data.kz[p, m] * bead_center[2])
                ) / farfield_data.kz[p, m]

                Ex[:, :, :] += np.reshape(
                    E[0, :], local_coordinates.coordinate_shape)
                Ey[:, :, :] += np.reshape(
                    E[1, :], local_coordinates.coordinate_shape)
                Ez[:, :, :] += np.reshape(
                    E[2, :], local_coordinates.coordinate_shape)

                if magnetic_field:
                    H[:] = 0
                    if internal:
                        H[:, region] = internal_H_field_fixed_r(
                            cn, dn,
                            internal_radial_data.sphBessel,
                            internal_radial_data.jn_over_k1r,
                            internal_radial_data.jn_1,
                            alp_expanded,
                            alp_sin_expanded, alp_deriv_expanded,
                            cosT, cosP, sinP, bead.n_bead
                        )
                    else:
                        H[:, region] = scattered_H_field_fixed_r(
                            an, bn,
                            external_radial_data.krH,
                            external_radial_data.dkrH_dkr,
                            external_radial_data.k0r,
                            alp_expanded, alp_sin_expanded,
                            alp_deriv_expanded, cosT, cosP, sinP,
                            bead.n_medium, total_field
                        )

                    H = np.matmul(A.T, H)
                    H[:, region] *= E0[polarization] * np.exp(1j * (
                        farfield_data.kx[p, m] * bead_center[0] +
                        farfield_data.ky[p, m] * bead_center[1] +
                        farfield_data.kz[p, m] * bead_center[2])
                    ) / farfield_data.kz[p, m]

                    Hx[:, :, :] += np.reshape(
                        H[0, :], local_coordinates.coordinate_shape)
                    Hy[:, :, :] += np.reshape(
                        H[1, :], local_coordinates.coordinate_shape)
                    Hz[:, :, :] += np.reshape(
                        H[2, :], local_coordinates.coordinate_shape)


@njit(cache=True)
def scattered_field_fixed_r(
    an: np.ndarray, bn: np.ndarray, krh: np.ndarray,
    dkrh_dkr: np.ndarray, k0r: np.ndarray,
    alp: np.ndarray, alp_sin: np.ndarray,
    alp_deriv: np.ndarray,
    cos_theta: np.ndarray, cosP: np.ndarray,
    sinP: np.ndarray,
    total_field=True
):

    # Radial, theta and phi-oriented fields
    Er = np.zeros((1, cos_theta.shape[0]), dtype='complex128')
    Et = np.zeros((1, cos_theta.shape[0]), dtype='complex128')
    Ep = np.zeros((1, cos_theta.shape[0]), dtype='complex128')

    sinT = np.sqrt(1 - cos_theta**2)  # for theta = 0...pi

    L = np.arange(start=1, stop=an.size + 1)
    C1 = 1j**(L + 1) * (2 * L + 1)
    C2 = C1 / (L * (L + 1))
    for L in range(an.size, 0, -1):
        Er += C1[L-1] * an[L - 1] * krh[L - 1, :] * alp[L-1, :]

        Et += C2[L-1] * (
            an[L - 1] * dkrh_dkr[L - 1, :] * alp_deriv[L - 1, :] +
            1j * bn[L - 1] * krh[L - 1, :] * alp_sin[L - 1, :]
        )

        Ep += C2[L-1] * (
            an[L - 1] * dkrh_dkr[L - 1, :] * alp_sin[L - 1, :] +
            1j * bn[L - 1] * krh[L - 1, :] * alp_deriv[L - 1, :]
        )

    Er *= -cosP / (k0r)**2
    Et *= -cosP / (k0r)
    Ep *= sinP / (k0r)
    # Cartesian components
    Ex = Er * sinT * cosP + Et * cos_theta * cosP - Ep * sinP
    Ey = Er * sinT * sinP + Et * cos_theta * sinP + Ep * cosP
    Ez = Er * cos_theta - Et * sinT
    if total_field:
        # Incident field (x-polarized)
        Ei = np.exp(1j * k0r * cos_theta)
        return np.concatenate((Ex + Ei, Ey, Ez), axis=0)
    else:
        return np.concatenate((Ex, Ey, Ez), axis=0)


@njit(cache=True)
def internal_field_fixed_r(
    cn: np.ndarray, dn: np.ndarray,
    sphBessel: np.ndarray, jn_over_k1r: np.ndarray, jn_1: np.ndarray,
    alp: np.ndarray, alp_sin: np.ndarray, alp_deriv: np.ndarray,
    cos_theta: np.ndarray, cosP: np.ndarray, sinP: np.ndarray
):

    # Radial, theta and phi-oriented fields
    Er = np.zeros((1, cos_theta.shape[0]), dtype='complex128')
    Et = np.zeros((1, cos_theta.shape[0]), dtype='complex128')
    Ep = np.zeros((1, cos_theta.shape[0]), dtype='complex128')

    sinT = np.sqrt(1 - cos_theta**2)

    for n in range(cn.size, 0, -1):
        Er += - (1j**(n + 1) * (2*n + 1) * alp[n - 1, :] * dn[n - 1] *
                 jn_over_k1r[n - 1, :])

        Et += 1j**n * (2 * n + 1) / (n * (n + 1)) * (
            cn[n - 1] *
            alp_sin[n - 1, :] * sphBessel[n - 1, :] - 1j * dn[n - 1] *
            alp_deriv[n - 1, :] * (
                jn_1[n - 1, :] - n * jn_over_k1r[n - 1, :]
            )
        )

        Ep += -1j**n * (2 * n + 1) / (n * (n + 1)) * (
            cn[n - 1] * alp_deriv[n - 1, :] * sphBessel[n - 1, :] -
            1j*dn[n - 1] * alp_sin[n - 1, :] * (
                jn_1[n - 1, :] - n * jn_over_k1r[n - 1, :])
        )

    Er *= -cosP
    Et *= -cosP
    Ep *= -sinP
    # Cartesian components
    Ex = Er * sinT * cosP + Et * cos_theta * cosP - Ep * sinP
    Ey = Er * sinT * sinP + Et * cos_theta * sinP + Ep * cosP
    Ez = Er * cos_theta - Et * sinT
    return np.concatenate((Ex, Ey, Ez), axis=0)


@njit(cache=True)
def scattered_H_field_fixed_r(
    an: np.ndarray, bn: np.ndarray, krh: np.ndarray,
    dkrh_dkr: np.ndarray, k0r: np.ndarray,
    alp: np.ndarray, alp_sin: np.ndarray,
    alp_deriv: np.ndarray,
    cos_theta: np.ndarray, cosP: np.ndarray,
    sinP: np.ndarray, n_medium: float,
    total_field=True
):

    # Radial, theta and phi-oriented fields
    Hr = np.zeros((1, cos_theta.shape[0]), dtype='complex128')
    Ht = np.zeros((1, cos_theta.shape[0]), dtype='complex128')
    Hp = np.zeros((1, cos_theta.shape[0]), dtype='complex128')

    sinT = np.sqrt(1 - cos_theta**2)

    L = np.arange(start=1, stop=an.size + 1)
    C1 = 1j**L * (2 * L + 1)
    C2 = C1 / (L * (L + 1))
    for L in range(an.size, 0, -1):
        Hr += C1[L - 1] * 1j * bn[L - 1] * krh[L - 1, :] * alp[L - 1, :]

        Ht += C2[L - 1] * (
            1j * bn[L - 1] * dkrh_dkr[L - 1, :] * alp_deriv[L - 1, :] -
            an[L - 1] * krh[L - 1, :] * alp_sin[L - 1, :]
        )

        Hp += C2[L - 1] * (
            1j * bn[L - 1] * dkrh_dkr[L - 1, :] * alp_sin[L - 1, :] -
            an[L - 1] * krh[L - 1, :] * alp_deriv[L - 1, :]
        )

    # Extra factor of -1 as B&H does not include the Condon–Shortley phase,
    # but our associated Legendre polynomials do include it
    Hr *= -sinP / (k0r)**2 * n_medium / (_C * _MU0)
    Ht *= -sinP / (k0r) * n_medium / (_C * _MU0)
    Hp *= -cosP / (k0r) * n_medium / (_C * _MU0)

    # Cartesian components
    Hx = Hr * sinT * cosP + Ht * cos_theta * cosP - Hp * sinP
    Hy = Hr * sinT * sinP + Ht * cos_theta * sinP + Hp * cosP
    Hz = Hr * cos_theta - Ht * sinT
    if total_field:
        # Incident field (E field x-polarized)
        Hi = np.exp(1j * k0r * cos_theta) * n_medium / (_C * _MU0)
        return np.concatenate((Hx, Hy + Hi, Hz), axis=0)
    else:
        return np.concatenate((Hx, Hy, Hz), axis=0)


@njit(cache=True)
def internal_H_field_fixed_r(
    cn: np.ndarray, dn: np.ndarray,
    sphBessel: np.ndarray, jn_over_k1r: np.ndarray, jn_1: np.ndarray,
    alp: np.ndarray, alp_sin: np.ndarray, alp_deriv: np.ndarray,
    cos_theta: np.ndarray, cosP: np.ndarray, sinP: np.ndarray,
    n_bead: np.complex128
):

    # Radial, theta and phi-oriented fields
    Hr = np.zeros((1, cos_theta.shape[0]), dtype='complex128')
    Ht = np.zeros((1, cos_theta.shape[0]), dtype='complex128')
    Hp = np.zeros((1, cos_theta.shape[0]), dtype='complex128')

    sinT = np.sqrt(1 - cos_theta**2)

    for n in range(cn.size, 0, -1):
        Hr += (1j**(n + 1) * (2*n + 1) * alp[n - 1, :] * cn[n - 1] *
               jn_over_k1r[n - 1, :])

        Ht += 1j**n * (2 * n + 1) / (n * (n + 1)) * (
            dn[n - 1] * alp_sin[n - 1, :] * sphBessel[n - 1, :] -
            1j * cn[n - 1] * alp_deriv[n - 1, :] * (
                jn_1[n - 1, :] - n * jn_over_k1r[n - 1, :]
            )
        )

        Hp += - 1j**n * (2 * n + 1) / (n * (n + 1)) * (
            dn[n - 1] * alp_deriv[n - 1, :] * sphBessel[n - 1, :] -
            1j*cn[n - 1] * alp_sin[n - 1, :] * (
                jn_1[n - 1, :] - n * jn_over_k1r[n - 1, :]
            )
        )

    Hr *= sinP * n_bead / (_C * _MU0)
    Ht *= -sinP * n_bead / (_C * _MU0)
    Hp *= cosP * n_bead / (_C * _MU0)
    # Cartesian components
    Hx = Hr * sinT * cosP + Ht * cos_theta * cosP - Hp * sinP
    Hy = Hr * sinT * sinP + Ht * cos_theta * sinP + Hp * cosP
    Hz = Hr * cos_theta - Ht * sinT
    return np.concatenate((Hx, Hy, Hz), axis=0)


def R_phi(cos_phi, sin_phi):
    return np.asarray([[cos_phi, -sin_phi, 0],
                       [sin_phi, cos_phi, 0],
                       [0, 0, 1]])


def R_th(cos_th, sin_th):
    return np.asarray([[cos_th, 0, sin_th],
                       [0,  1,  0],
                       [-sin_th, 0, cos_th]])

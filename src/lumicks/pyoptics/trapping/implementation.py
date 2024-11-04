import numpy as np
from numba import njit
from scipy.constants import mu_0 as MU0, speed_of_light as C


@njit(cache=True, parallel=False)
def scattered_field_fixed_r(
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
def internal_field_fixed_r(
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
def scattered_H_field_fixed_r(
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
def internal_H_field_fixed_r(
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


def R_phi(cos_phi: float, sin_phi: float):
    return np.asarray([[cos_phi, -sin_phi, 0], [sin_phi, cos_phi, 0], [0, 0, 1]])


def R_th(cos_th: float, sin_th: float):
    return np.asarray([[cos_th, 0, sin_th], [0, 1, 0], [-sin_th, 0, cos_th]])


@njit
def R_th_R_phi(cos_theta: float, sin_theta: float, cos_phi: float, sin_phi: float):
    return np.asarray(
        [
            [cos_phi * cos_theta, -cos_theta * sin_phi, sin_theta],
            [sin_phi, cos_phi, 0],
            [-cos_phi * sin_theta, sin_phi * sin_theta, cos_theta],
        ]
    )


@njit
def R_pol_R_th_R_phi(cos_theta: float, sin_theta: float, cos_phi: float, sin_phi: float):
    return np.asarray(
        [
            [sin_phi, cos_phi, 0],
            [-cos_phi * cos_theta, cos_theta * sin_phi, -sin_theta],
            [-cos_phi * sin_theta, sin_phi * sin_theta, cos_theta],
        ]
    )

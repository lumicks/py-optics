"""Field distributions of dipoles, implemented from various sources"""

import numpy as np
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0
from scipy.constants import speed_of_light as C

from ..mathutils.vector import cosines_from_unit_vectors, spherical_to_cartesian

def field_dipole_x(px, n_medium, lambda_vac, x, y, z):
    """Get the electromagnetic field of an x-oriented dipole in homogeneous space. The field includes
    both near- and farfields. The dipole is located at (0,0,0). See [1]_.

    Parameters
    ----------
    px : float
        Dipole moment of the dipole (SI units)
    n_medium : float
        Refractive index of the medium in which the dipole is embedded
    lambda_vac : float
        Wavelength in vacuum of the radiation
    x, y, z : Union[float, np.ndarray]
        (Array of) coordinates at which the electromagnetic field is to be evaluated

    Returns
    -------
    Ex : np.ndarray
        Array of electric field polarized in the x-direction evaluated at (x, y, z)
    Ey : np.ndarray
        As Ex, but y-polarized component
    Ez : np.ndarray
        As Ex, but z-polarized component
    Hx : np.ndarray
        H field polarized in the x-direction evaluated at (x, y, z)
    Hy : np.ndarray
        As Hx, but y-polarized component
    Hz : np.ndarray
        As Hx, but z-polarized component


    ..  [1] Principles of Nano-optics, 2nd Ed., Ch. 2
    """
    x, y, z = [np.atleast_1d(ax) for ax in (x, y, z)]
    k = 2 * np.pi * n_medium / lambda_vac
    R = np.hypot(np.hypot(x, y), z)

    fix = R == 0
    R[fix] = 1

    Ex = 1 + 1 / (k**2 * R) * (
        -(k**2) * x**2 / R + (2 * x**2 - y**2 - z**2) * (1 / R**3 - 1j * k / R**2)
    )
    Ey = x * y / (k**2 * R) * (3 / R**3 - 3j * k / R**2 - k**2 / R)
    Ez = x * z / (k**2 * R) * (3 / R**3 - 3j * k / R**2 - k**2 / R)

    prefactor = px * k**2 * np.exp(1j * k * R) / (4 * np.pi * R * EPS0 * n_medium**2)
    Ex *= prefactor
    Ey *= prefactor
    Ez *= prefactor

    Hy = 1j * k * z / R - z / R**2
    Hz = y / R**2 - 1j * k * y / R

    prefactor *= (1j * 2 * np.pi * C / lambda_vac * MU0) ** -1

    Hy *= prefactor
    Hz *= prefactor

    Hx = np.zeros(Hy.shape, dtype="complex128")

    Ex[fix] = Ey[fix] = Ez[fix] = Hx[fix] = Hy[fix] = Hz[fix] = np.nan + 1j * np.nan

    return Ex, Ey, Ez, Hx, Hy, Hz


def field_dipole_y(py, n_medium, lambda_vac, x, y, z):
    """Get the electromagnetic field of a y-oriented dipole. The field includes
    both near- and farfields. The implementation is based on a permutation of the fields as
    calculated by :func:`field_dipole_z()`.

    Parameters
    ----------

    py : float
        Dipole moment of the dipole (SI units)
    n_medium : float
        Refractive index of the medium in which the dipole is embedded
    lambda_vac : float
        Wavelength in vacuum of the radiation
    x, y, z : Union[float, np.ndarray]
        (Array of) coordinates at which the electromagnetic field is to be evaluated

    Returns
    -------
    Ex : np.ndarray
        Array of electric field polarized in the x-direction evaluated at (x, y, z)
    Ey : np.ndarray
        As Ex, but y-polarized component
    Ez : np.ndarray
        As Ex, but z-polarized component
    Hx : np.ndarray
        H field polarized in the x-direction evaluated at (x, y, z)
    Hy : np.ndarray
        As Hx, but y-polarized component
    Hz : np.ndarray
        As Hx, but z-polarized component
    """

    Ex, Ez, Ey, Hx, Hz, Hy = field_dipole_z(py, n_medium, lambda_vac, x, -z, y)
    Ez *= -1
    Hz *= -1
    return Ex, Ey, Ez, Hx, Hy, Hz


def field_dipole_z(pz, n_medium, lambda_vac, x, y, z):
    """Get the electromagnetic field of a z-oriented dipole. The field includes
    both near- and farfields. The dipole is located at (0,0,0). See [1]_

    Parameters
    ----------
    pz : float
        Dipole moment of the dipole (SI units)
    n_medium : float
        Refractive index of the medium in which the dipole is embedded
    lambda_vac : float
        wavelength in vacuum of the radiation
    x, y, z : np.ndarray
        (Array of) coordinates at which the electromagnetic field is to be evaluated

    Returns
    -------
    Ex : np.ndarray
        Array of electric field polarized in the x-direction evaluated at (x, y, z)
    Ey : np.ndarray
        As Ex, but y-polarized component
    Ez : np.ndarray
        As Ex, but z-polarized component
    Hx : np.ndarray
        H field polarized in the x-direction evaluated at (x, y, z)
    Hy : np.ndarray
        As Hx, but y-polarized component
    Hz : np.ndarray
        As Hx, but z-polarized component


    ..  [1] Antenna Theory, Ch. 4, 3rd Edition, C. A. Balanis
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    rho = np.hypot(x, y)
    r = np.hypot(rho, z)

    fix = r == 0
    r[fix] = 1
    cosT = z / r
    sinT = (1 - cosT**2) ** 0.5

    # Division by zero when x == y == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        cosP = x / rho
        sinP = y / rho

    # based on physics/symmetry, the field is purely z-oriented whenever x == y
    # == 0. Therefore we can safely set the cosP and sinP to zero in that case
    # as it does not affect Ez (no cosP or sinP terms)
    cosP[rho == 0] = sinP[rho == 0] = 0

    _eps = EPS0 * n_medium**2
    eta = (MU0 / _eps) ** 0.5
    k = 2 * np.pi * n_medium / lambda_vac
    w = C * (k / n_medium)

    I0l = 1j * w * pz

    Er = (eta * I0l * cosT) / (2 * np.pi * r**2) * (1 + 1 / (1j * k * r)) * np.exp(-1j * k * r)
    Et = (
        (1j * eta * k * I0l * sinT)
        / (4 * np.pi * r)
        * (1 + (1j * k * r) ** -1 - (k * r) ** -2)
        * np.exp(-1j * k * r)
    )

    Hp = (1j * k * I0l * sinT) / (4 * np.pi * r) * (1 + (1j * k * r) ** -1) * np.exp(-1j * k * r)

    Ex = Er * sinT * cosP + Et * cosT * cosP
    Ey = Er * sinT * sinP + Et * cosT * sinP
    Ez = Er * cosT - Et * sinT

    Hx = -Hp * sinP
    Hy = Hp * cosP
    Hz = np.zeros(Hp.shape, dtype="complex128")

    # Balanis uses exp(1j * w * t) as a convention, we use exp(-1j * w * t)
    Ex = np.conj(Ex)
    Ey = np.conj(Ey)
    Ez = np.conj(Ez)

    Hx = np.conj(Hx)
    Hy = np.conj(Hy)

    Ex[fix] = Ey[fix] = Ez[fix] = Hx[fix] = Hy[fix] = Hz[fix] = np.nan + 1j * np.nan

    return Ex, Ey, Ez, Hx, Hy, Hz


def field_dipole(p, n_medium, lambda_vac, x, y, z, farfield=False):
    """Get the electromagnetic field of an arbitrarily-oriented dipole.
    The field includes both near- and farfields. The dipole is located at (0,0,0) See [1]_. This
    function was not tested with complex dipole moment components.

    Parameters
    ----------
    p : tuple(float, float, float)
        The dipole moment (px, py, pz) of the dipole (SI units)
    n_medium : float
        Refractive index of the medium in which the dipole is embedded
    lambda_vac : float
        Wavelength in vacuum of the radiation
    x, y, z : np.ndarray
        (Array of) coordinates at which the electromagnetic field is to be evaluated

    Returns
    -------
    Ex : np.ndarray
        Array of electric field polarized in the x-direction evaluated at (x, y, z)
    Ey : np.ndarray
        As Ex, but y-polarized component
    Ez : np.ndarray
        As Ex, but z-polarized component
    Hx : np.ndarray
        H field polarized in the x-direction evaluated at (x, y, z)
    Hy : np.ndarray
        As Hx, but y-polarized component
    Hz : np.ndarray
        As Hx, but z-polarized component


    ..  [1] Classical Electrodynamics, Ch. 9, 3rd Edition, J.D. Jackson
    """
    x, y, z = [np.atleast_1d(ax) for ax in (x, y, z)]

    r = np.hypot(np.hypot(x, y), z)
    k = 2 * np.pi * n_medium / lambda_vac
    fix = r == 0
    r[fix] = 1
    nx, ny, nz = [ax / r for ax in (x, y, z)]
    px, py, pz = p

    nxp_x = ny * pz - nz * py
    nxp_y = nz * px - nx * pz
    nxp_z = nx * py - ny * px

    nxpxn_x = nxp_y * nz - nxp_z * ny
    nxpxn_y = nxp_z * nx - nxp_x * nz
    nxpxn_z = nxp_x * ny - nxp_y * nx

    ndotp = nx * px + ny * py + nz * pz

    eps_inv = (EPS0 * n_medium**2) ** -1
    G0 = np.exp(1j * k * r) / (4 * np.pi * r)

    if farfield:
        Ex = k**2 * nxpxn_x + 0j
        Ey = k**2 * nxpxn_y + 0j
        Ez = k**2 * nxpxn_z + 0j
        Hx = np.ones(Ex.shape, dtype="complex128")
    else:
        Ex = k**2 * nxpxn_x + (3 * nx * ndotp - px) * (r**-2 - 1j * k / r)
        Ey = k**2 * nxpxn_y + (3 * ny * ndotp - py) * (r**-2 - 1j * k / r)
        Ez = k**2 * nxpxn_z + (3 * nz * ndotp - pz) * (r**-2 - 1j * k / r)
        Hx = 1 - (1j * k * r) ** -1
    Hy = Hx.copy()
    Hz = Hx.copy()

    Ex *= G0 * eps_inv
    Ey *= G0 * eps_inv
    Ez *= G0 * eps_inv

    prefactor_H = G0 * C * k**2 / n_medium

    Hx *= prefactor_H * nxp_x
    Hy *= prefactor_H * nxp_y
    Hz *= prefactor_H * nxp_z

    Ex[fix] = Ey[fix] = Ez[fix] = Hx[fix] = Hy[fix] = Hz[fix] = np.nan + 1j * np.nan

    return Ex, Ey, Ez, Hx, Hy, Hz


def farfield_dipole_position(p, n_medium, lambda_vac, x, y, z):
    """Get the electric farfield of an arbitrarily-oriented dipole.
    The dipole is located at (0,0,0).
    See [1]_

    Parameters
    ----------
    p : tuple(float, float, float)
        The dipole moment (px, py, pz) of the dipole (SI units)
    n_medium : float
        Refractive index of the medium in which the dipole is embedded
    lambda_vac : float
        Wavelength in vacuum of the radiation
    x, y, z : Union[float, np.ndarray]
        (Array of locations in the far field where to evaluate the electric field (meters).

    Returns
    -------
    Ex : np.ndarray
        Array of electric field polarized in the x-direction evaluated at (x, y, z)
    Ey : np.ndarray
        As Ex, but y-polarized component
    Ez : np.ndarray
        As Ex, but z-polarized component


    ..  [1] Principles of Nano-optics, 2nd Ed., Appendix D
    """
    x, y, z = [np.atleast_1d(ax) for ax in (x, y, z)]
    if not (x.shape == y.shape == z.shape):
        raise ValueError("Location coordinates x, y and z need to be of the same size or shape")
    r = np.hypot(x, np.hypot(y, z))
    sx, sy, sz = [ax / r for ax in (x, y, z)]
    cos_theta, sin_theta, cos_phi, sin_phi = cosines_from_unit_vectors(sx, sy, sz)
    Ex, Ey, Ez = farfield_dipole_angle(
        p, n_medium, lambda_vac, cos_phi, sin_phi, cos_theta, sin_theta, r
    )
    return Ex, Ey, Ez


def farfield_dipole_angle(p, n_medium, lambda_vac, cos_phi, sin_phi, cos_theta, sin_theta, r):
    """Get the electromagnetic farfield of an arbitrarily-oriented dipole.
    The dipole is located at (0,0,0). See [1]_.

    Parameters
    ----------
    p : tuple(float, float, float)
        Tuple of (px, py, pz), the dipole moment of the dipole (SI units)
    n_medium : float
        Refractive index of the medium in which the dipole is embedded
    lambda_vac : float
        Wavelength in vacuum of the radiation (meters)
    cos_phi : Union[float, np.ndarray]
        Cosine of the angle phi, which is the angle between the location (x, y, 0) and the x-axis.
    sin_phi : Union[float, np.ndarray]
        Sine of the angle phi
    cos_theta : Union[float, np.ndarray]
        Cosine of the angle Theta, which is the angle of the location (x, y, z) with the z-axis.
    r: distance from (0, 0, 0) to (x, y, z)

    Returns
    -------
    Ex: np.ndarray
        Array of electric field polarized in the x-direction evaluated at (x, y, z)
    Ey: np.ndarray
        As Ex, but y-polarized component
    Ez: np.ndarray
        As Ex, but z-polarized component


    ..  [1] Principles of Nano-optics, 2nd Ed., Appendix D
    """
    cos_phi = np.atleast_1d(cos_phi)
    sin_phi = np.atleast_1d(sin_phi)
    cos_theta = np.atleast_1d(cos_theta)

    assert cos_phi.shape == sin_phi.shape == cos_theta.shape

    sinT = np.zeros(cos_theta.shape)
    sinT[cos_theta <= 1] = (1 - cos_theta[cos_theta <= 1] ** 2) ** 0.5

    k = 2 * np.pi * n_medium / lambda_vac
    prefactor = k**2 * np.exp(1j * k * r) / (n_medium**2 * EPS0 * 4 * np.pi * r)
    Ex = (
        p[0] * (1 - cos_phi**2 * sinT**2)
        - p[1] * sin_phi * cos_phi * sinT**2
        - p[2] * cos_phi * sinT * cos_theta
    ) * prefactor
    Ey = (
        -p[0] * sin_phi * cos_phi * sinT**2
        + p[1] * (1 - sin_phi**2 * sinT**2)
        - p[2] * sin_phi * sinT * cos_theta
    ) * prefactor
    Ez = (
        -p[0] * cos_phi * sinT * cos_theta - p[1] * sin_phi * sinT * cos_theta + p[2] * sinT**2
    ) * prefactor

    return Ex, Ey, Ez

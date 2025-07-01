"""Test full 3D vector point spread function of focused dipoles against approximate (paraxial)
expressions, for low NA objectives."""

import numpy as np
import pytest
import scipy.special as sp
from numpy.typing import ArrayLike
from scipy.constants import epsilon_0

import lumicks.pyoptics.psf.reference as ref


def focused_dipole_paraxial_xy(
    dipole_moment_xy: float,
    lambda_vac: float,
    n_image: float,
    n_medium: float,
    focal_length: float,
    NA: float,
    focal_length_tube: float,
    r: ArrayLike,
) -> np.ndarray:
    """Calculate the point spread function of a focused dipole in the xy-plane,
    in the paraxial approximation. In that case, the point spread function for x- and y-oriented
    dipoles is the same.

    Parameters
    ----------
    dipole_moment_xy : float
        dipole moment in [Cm]
    lambda_vac : float
        wavelength [m]
    n_image : float
        Refractive index of the medium at the focal plane of the tube lens [-].
    n_medium : float
        Refractive index of the medium that the dipole resides in [-].
    focal_length : float
        focal length of the objective [m].
    NA : float
        Numerical aperture of the objective
    focal_length_tube : float
        focal length of the tube lens [m].
    r : np.array
        radial distance [m].

    Returns
    -------
    squared_electric_field : np.ndarray
        The squared magnitude of the electric field in [V^2/m^2], for dipoles oriented in the plane.
    """
    M = focal_length_tube / focal_length * n_medium / n_image
    r_ = NA * np.asarray(r) / (M * lambda_vac) * 2 * np.pi

    with np.errstate(divide="ignore", invalid="ignore"):
        squared_electric_field = (2 * sp.jv(1, r_) / r_) ** 2
    squared_electric_field[r_ == 0] = 1.0
    squared_electric_field *= (dipole_moment_xy**2 * (NA * np.pi) ** 4) / (
        epsilon_0**2 * n_medium * n_image * lambda_vac**6 * M**2
    )

    return squared_electric_field


def focused_dipole_paraxial_z(
    dipole_moment_z: float,
    lambda_vac: float,
    n_bfp: float,
    n_medium: float,
    focal_length: float,
    NA: float,
    focal_length_tube: float,
    r: ArrayLike,
) -> np.ndarray:
    """Calculate the point spread function of a focused dipole oriented along the z-axis,
    in the paraxial approximation.

    Parameters
    ----------
    dipole_moment_z : float
        dipole moment in [Cm]
    lambda_vac : float
        wavelength [m]
    n_image : float
        Refractive index of the medium at the focal plane of the tube lens [-].
    n_medium : float
        Refractive index of the medium that the dipole resides in [-].
    focal_length : float
        focal length of the objective [m].
    NA : float
        Numerical aperture of the objective
    focal_length_tube : float
        focal length of the tube lens [m].
    r : np.array
        radial distance [m].

    Returns
    -------
    squared_electric_field : np.ndarray
        The squared magnitude of the electric field [V^2/m^2]
    """
    M = focal_length_tube / focal_length * n_medium / n_bfp
    r_ = NA * np.asarray(r) / (M * lambda_vac) * 2 * np.pi

    with np.errstate(divide="ignore", invalid="ignore"):
        squared_electric_field = (2 * sp.jv(2, r_) / r_) ** 2
    squared_electric_field[r_ == 0] = 0.0
    squared_electric_field *= (dipole_moment_z**2 * (NA / lambda_vac) ** 6 * np.pi**4) / (
        epsilon_0**2 * n_medium**3 * n_bfp * M**2
    )
    return squared_electric_field


@pytest.mark.parametrize("dipole_moment", [1e-30, 3e-29])
@pytest.mark.parametrize("wavelength", [680e-9, 525e-9])
@pytest.mark.parametrize("n_image, n_medium", [(1.0, 1.0), (1.33, 1.5)])
@pytest.mark.parametrize("NA", [0.025, 0.005])
@pytest.mark.parametrize("focal_length_tube", [180e-3, 200e-3])
@pytest.mark.parametrize("magnification", [10.0, 60.0])
def test_paraxial_psf_xy(
    dipole_moment, wavelength, n_image, n_medium, NA, focal_length_tube, magnification
):
    p = dipole_moment
    num_points = 27
    range = 0.61 * wavelength / NA * 5.0
    f_ = focal_length_tube
    f = f_ / magnification * n_medium
    x = np.linspace(-range * 0.5, range * 0.5, num_points) * magnification

    Esq = focused_dipole_paraxial_xy(
        p,
        wavelength,
        n_image=n_image,
        n_medium=n_medium,
        focal_length=f,
        NA=NA,
        focal_length_tube=focal_length_tube,
        r=x,
    )
    Ex_x, Ey_x = ref.focused_dipole_ref(
        [p, 0, 0],
        wavelength,
        n_image=n_image,
        n_medium=n_medium,
        focal_length=f,
        NA=NA,
        focal_length_tube=focal_length_tube,
        x=x,
        y=x,
        z=0,
    )

    Ex_y, Ey_y = ref.focused_dipole_ref(
        [0, p, 0],
        wavelength,
        n_image=n_image,
        n_medium=n_medium,
        focal_length=f,
        NA=NA,
        focal_length_tube=focal_length_tube,
        x=x,
        y=x,
        z=0,
    )
    Idip_x_cols = (np.abs(Ex_x) ** 2 + np.abs(Ey_x) ** 2)[(num_points - 1) // 2, :]
    Idip_x_rows = (np.abs(Ex_x) ** 2 + np.abs(Ey_x) ** 2)[:, (num_points - 1) // 2]

    Idip_y_cols = (np.abs(Ex_y) ** 2 + np.abs(Ey_y) ** 2)[(num_points - 1) // 2, :]
    Idip_y_rows = (np.abs(Ex_y) ** 2 + np.abs(Ey_y) ** 2)[:, (num_points - 1) // 2]

    # Paraxial PSF should match that of an exact PSF for x-oriented dipole
    np.testing.assert_allclose(Esq, Idip_x_cols, rtol=1e-2, atol=0)
    np.testing.assert_allclose(Esq, Idip_x_rows, rtol=1e-2, atol=0)
    # Paraxial PSF should match that of an exact PSF for y-oriented dipole
    np.testing.assert_allclose(Esq, Idip_y_cols, rtol=1e-2, atol=0)
    np.testing.assert_allclose(Esq, Idip_y_rows, rtol=1e-2, atol=0)


@pytest.mark.parametrize("dipole_moment", [1e-30, 3e-29])
@pytest.mark.parametrize("wavelength", [680e-9, 525e-9])
@pytest.mark.parametrize("n_bfp, n_medium", [(1.0, 1.0), (1.33, 1.5)])
@pytest.mark.parametrize("NA", [0.025, 0.005])
@pytest.mark.parametrize("focal_length_tube", [180e-3, 200e-3])
@pytest.mark.parametrize("magnification", [10.0, 60.0])
def test_paraxial_psf_z(
    dipole_moment, wavelength, NA, n_bfp, n_medium, focal_length_tube, magnification
):
    p = dipole_moment
    num_points = 27
    range = 0.61 * wavelength / NA * 5.0
    f_ = focal_length_tube
    f = f_ / magnification * n_medium
    x = np.linspace(-range * 0.5, range * 0.5, num_points) * magnification

    Esq = focused_dipole_paraxial_z(
        p,
        wavelength,
        n_bfp=n_bfp,
        n_medium=n_medium,
        focal_length=f,
        NA=NA,
        focal_length_tube=focal_length_tube,
        r=x,
    )
    Ex_z, Ey_z = ref.focused_dipole_ref(
        [0, 0, p],
        wavelength,
        n_bfp,
        n_medium,
        focal_length=f,
        NA=NA,
        focal_length_tube=focal_length_tube,
        x=x,
        y=x,
        z=0,
    )

    Idip_z_cols = (np.abs(Ex_z) ** 2 + np.abs(Ey_z) ** 2)[(num_points - 1) // 2, :]
    Idip_z_rows = (np.abs(Ex_z) ** 2 + np.abs(Ey_z) ** 2)[:, (num_points - 1) // 2]

    np.testing.assert_allclose(Esq, Idip_z_cols, rtol=1e-2, atol=0)
    np.testing.assert_allclose(Esq, Idip_z_rows, rtol=1e-2, atol=0)

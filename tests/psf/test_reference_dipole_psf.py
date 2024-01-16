"""Test full 3D vector point spread function of focused dipoles against approximate (paraxial)
expressions, for low NA objectives."""

import numpy as np
import pytest

import lumicks.pyoptics.psf.reference as ref


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

    Esq = ref.focused_dipole_paraxial_xy(
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

    Esq = ref.focused_dipole_paraxial_z(
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

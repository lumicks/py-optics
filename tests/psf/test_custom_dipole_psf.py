import numpy as np
import pytest

import lumicks.pyoptics.farfield_transform as tf
import lumicks.pyoptics.field_distributions as fd
import lumicks.pyoptics.psf as psf
import lumicks.pyoptics.psf.reference as ref
from lumicks.pyoptics.objective import Objective


def gen_dipole_psf(
    dipole_moment, lambda_em, NA_objective, n_medium, n_image_plane, f_tube_lens, magnification
):
    """Calculate the point spread function of a focused dipole. First calculate the far field, then
    use an Objective to transform that field to the back focal plane. Use another Objective with
    long focal length to focus the field on the image plane.

    Parameters
    ----------
    dipole_moment : Tuple[float, float, float]
        The x, y, and z parts of the dipole moment, in Cm
    lambda_em : float
        Emission wavelength, in meters
    NA_objective : float
        Numerical aperture of the objective
    n_medium : float
        Refractive index of the immersion medium of the objective
    n_image_plane : float
        Refractive index of the material at the image plane
    f_tube_lens : float
        Focal length of the tube lens, in meters, as it would be in air.
    magnification : float
        Magnification of the microscope.

    Returns
    -------
    Ex, Ey, Ez : np.ndarray
        Electic field components of the dipole image, in V/m.
    """
    numpoints = 9
    bfp_sampling_n = 50

    f_ = f_tube_lens * n_image_plane
    f = f_tube_lens / magnification * n_medium
    NA_tube = n_image_plane * NA_objective / n_medium * f / f_

    xy_range = 2 * lambda_em
    dim_xy = (-xy_range / 2, xy_range / 2)
    z = np.linspace(dim_xy[0], dim_xy[1], numpoints)

    def dummy(_, x_bfp, *args):
        """A do-nothing function"""
        return (np.zeros_like(x_bfp), None)

    obj = Objective(
        NA=NA_objective,
        focal_length=f,
        n_bfp=1.0,
        n_medium=n_medium,
    )
    coords, fields = obj.sample_back_focal_plane(dummy, bfp_sampling_n)
    ff = obj.back_focal_plane_to_farfield(
        coords, fields, 1.0  # wavelength doesn't matter as we don't use kx, ky, kz
    )
    ff.kx = ff.ky = ff.kz = 0  # make that explicit

    def field_func(aperture, x_bfp, y_bfp, r_bfp, r_max, _):
        Ex, Ey, Ez = fd.dipole.electric_dipole_farfield_angle(
            dipole_moment,
            n_medium,
            lambda_em,
            ff.cos_phi,
            ff.sin_phi,
            ff.cos_theta,
            ff.sin_theta,
            obj.focal_length,
        )
        Ex_bfp, Ey_bfp = tf.ff_to_bfp_angle(
            Ex, Ey, Ez, ff.cos_phi, ff.sin_phi, ff.cos_theta, n_medium, n_bfp=1.0
        )

        return (Ex_bfp, Ey_bfp)

    Ex, Ey, Ez, X, Y, Z = psf.fast_psf(
        field_func,
        lambda_em,
        1.0,
        n_image_plane,
        f_,
        NA_tube,
        x_range=dim_xy,
        numpoints_x=numpoints,
        y_range=dim_xy,
        numpoints_y=numpoints,
        z=z * (n_image_plane / n_medium * magnification**2),
        bfp_sampling_n=bfp_sampling_n,
        return_grid=True,
    )

    return Ex, Ey, Ez, X, Y, Z


@pytest.mark.parametrize("dipole_moment", [(1e-30, 0, 0), (0, 1e-30, 0), (0, 0, 1e-30)])
@pytest.mark.parametrize("lambda_em", [680e-9, 450e-9])
@pytest.mark.parametrize(
    "NA_objective, n_medium, n_image_plane", [(0.45, 1.0, 1.0), (1.2, 1.33, 1.0), (1.49, 1.5, 1.33)]
)
@pytest.mark.parametrize("f_tube_lens", [200e-3, 180e-3])
@pytest.mark.parametrize("magnification", [100.0, 60.0])
def test_dipole_psf(
    dipole_moment, lambda_em, NA_objective, n_medium, n_image_plane, f_tube_lens, magnification
):
    Ex, Ey, _, X, Y, Z = gen_dipole_psf(
        dipole_moment, lambda_em, NA_objective, n_medium, n_image_plane, f_tube_lens, magnification
    )
    x = np.unique(X)
    y = np.unique(Y)
    z = np.unique(Z)

    Ex_ref, Ey_ref = ref.focused_dipole_ref(
        dipole_moment,
        lambda_em,
        n_image_plane,
        n_medium,
        f_tube_lens / magnification * n_medium,
        NA_objective,
        f_tube_lens * n_image_plane,
        x,
        y,
        z,
        False,
    )
    rtol = 2.0e-2
    atol = np.amax(np.abs(Ex_ref)) * rtol
    np.testing.assert_allclose(Ex, -Ex_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(Ey, -Ey_ref, rtol=rtol, atol=atol)

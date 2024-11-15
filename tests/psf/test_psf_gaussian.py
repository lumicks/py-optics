"""Test the discrete version of a focused Gaussian beam against a reference implementation"""

import numpy as np
import pytest
import scipy.special as sp

from lumicks.pyoptics.psf import fast_gauss
from lumicks.pyoptics.psf.reference import focused_gauss_ref


def paraxial_focus_gaussian(
    lambda_vac: float,
    n_bfp: float,
    n_medium: float,
    focal_length: float,
    NA: float,
    x: np.array,
    y: np.array,
):
    x, y = np.atleast_1d(x, y)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Calculate all (polar) distances r in the grid, as measured from (0,0)
    r_orig = np.hypot(X, Y)
    # Then transform the matrix into a vector
    r = np.reshape(r_orig, (1, -1))

    # Now get the unique numbers in that vector, so we only calculate the
    # integral for unique distances r
    r, idx_r = np.unique(r, return_inverse=True)
    k = 2 * np.pi / lambda_vac * n_medium
    th_max = np.arcsin(NA / n_medium)

    r_ = r * k * th_max
    with np.errstate(divide="ignore", invalid="ignore"):
        efield = sp.jv(1, r_) / r_ + 0j
    efield[r == 0] = 0.5
    efield *= (-1j * (n_bfp / n_medium) ** 0.5 * k * focal_length * th_max**2) * np.exp(
        -1j * k * focal_length
    )
    sx = X.shape
    return np.reshape(efield[idx_r], sx)


@pytest.mark.parametrize("wavelength", [500e-9, 1064e-9])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
@pytest.mark.parametrize("n_bfp, n_medium", [(1.0, 1.0), (1.0, 1.33), (1.5, 1.0)])
@pytest.mark.parametrize("NA", [0.001, 0.005])
def test_paraxial_gaussian(wavelength, focal_length, n_bfp, n_medium, NA):
    lambda_vac = wavelength
    filling_factor = 1e6
    num_pts = 51
    range = 0.61 * wavelength / NA * 5
    x_points = np.linspace(-range / 2, range / 2, num_pts)
    y_points = x_points

    Ex_ref, _, __ = focused_gauss_ref(
        lambda_vac=lambda_vac,
        n_bfp=n_bfp,
        n_medium=n_medium,
        focal_length=focal_length,
        filling_factor=filling_factor,
        NA=NA,
        x=x_points,
        y=y_points,
        z=0,
    )

    E = paraxial_focus_gaussian(
        lambda_vac=lambda_vac,
        n_bfp=n_bfp,
        n_medium=n_medium,
        focal_length=focal_length,
        NA=NA,
        x=x_points,
        y=y_points,
    )

    # For low NA, the vectorial PSF valid for high NA should in the limit of NA == 0 be the paraxial
    # PSF. Slap a boundary of 5% on it for the above NAs
    np.testing.assert_allclose(Ex_ref, E, rtol=5e-2, atol=0)


@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize(
    "x_shift, y_shift, z_shift", [(0, 0, 0), (1e-6, 0.5e-6, -0.25e-6), (-300e-9, -500e-9, 1e-6)]
)
def test_gaussian(focal_length, n_medium, NA, x_shift, y_shift, z_shift):
    n_bfp = 1.0
    bfp_sampling_n = 250
    lambda_vac = 1064e-9
    filling_factor = 0.9
    num_pts = 10

    z_points = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts) + z_shift
    x_points = np.linspace(-lambda_vac, lambda_vac, num_pts) + x_shift
    y_points = np.linspace(-lambda_vac, lambda_vac, num_pts) + y_shift
    x_dim = (-lambda_vac + x_shift, lambda_vac + x_shift)
    y_dim = (-lambda_vac + y_shift, lambda_vac + y_shift)

    Ex_ref, Ey_ref, Ez_ref = focused_gauss_ref(
        lambda_vac=lambda_vac,
        n_bfp=n_bfp,
        n_medium=n_medium,
        focal_length=focal_length,
        filling_factor=filling_factor,
        NA=NA,
        x=x_points,
        y=y_points,
        z=z_points,
    )

    Ex, Ey, Ez = fast_gauss(
        lambda_vac=lambda_vac,
        n_bfp=n_bfp,
        n_medium=n_medium,
        focal_length=focal_length,
        NA=NA,
        filling_factor=filling_factor,
        x_range=x_dim,
        numpoints_x=num_pts,
        y_range=y_dim,
        numpoints_y=num_pts,
        z=z_points,
        bfp_sampling_n=bfp_sampling_n,
        return_grid=False,
    )

    # Allow 1 V/m absolute error and 5% relative error
    np.testing.assert_allclose([Ex_ref, Ey_ref, Ez_ref], [Ex, Ey, Ez], rtol=0.05, atol=1)

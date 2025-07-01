"""Test the fully vectorial reference implementation of a focused Gaussian beam against a paraxial
version of the PSF"""

import numpy as np
import pytest
import scipy.special as sp
from numpy.typing import ArrayLike

from lumicks.pyoptics.psf.reference import focused_gauss_ref


def paraxial_focus_gaussian(
    lambda_vac: float,
    n_bfp: float,
    n_medium: float,
    focal_length: float,
    NA: float,
    x: ArrayLike,
    y: ArrayLike,
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

    Ex_ref, Ey_ref, Ez_ref = focused_gauss_ref(
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
    np.testing.assert_allclose(Ey_ref, 0, atol=1.0)
    np.testing.assert_allclose(Ez_ref, 0, atol=1.0)

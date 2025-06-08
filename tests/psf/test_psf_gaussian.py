"""Test the discretized version of a vectorial focused Gaussian beam against a reference
implementation."""

import numpy as np
import pytest

from lumicks.pyoptics.psf.czt import Objective, focus_gaussian_czt
from lumicks.pyoptics.psf.reference import focused_gauss_ref


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
    objective = Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

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

    Ex, Ey, Ez = focus_gaussian_czt(
        objective,
        lambda_vac=lambda_vac,
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

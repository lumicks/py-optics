"""Test the CZT and 2D quadrature versions of a vectorial focused Gaussian beam against a reference
implementation."""

import numpy as np
import pytest

from lumicks.pyoptics.objective import Objective
from lumicks.pyoptics.psf.czt import focus_gaussian_czt
from lumicks.pyoptics.psf.quad import focus_gaussian_quad
from lumicks.pyoptics.psf.reference import focused_gauss_ref


@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize(
    "x_shift, y_shift, z_shift", [(0, 0, 0), (1e-6, 0.5e-6, -0.25e-6), (-300e-9, -500e-9, 1e-6)]
)
def test_against_gaussian_reference(focal_length, n_medium, NA, x_shift, y_shift, z_shift):
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

    Ex_q, Ey_q, Ez_q = focus_gaussian_quad(
        objective,
        lambda_vac=lambda_vac,
        filling_factor=filling_factor,
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

    # Check brute-force 2D quadrature over disk with 1D numerical integration result.
    #
    # Allow rtol = 1e-4, as the integration scheme decides automatically but precision could be
    # improved with higher-order integration schemes.
    np.testing.assert_allclose([Ex_ref, Ey_ref, Ez_ref], [Ex_q, Ey_q, Ez_q], rtol=1e-4)

    # Check CZT result with 1D numerical integration result.
    #
    # Allow 1 V/m absolute error and 5% relative error, as the CZT version converges relatively
    # slowly
    np.testing.assert_allclose([Ex_ref, Ey_ref, Ez_ref], [Ex, Ey, Ez], rtol=0.05, atol=1)


@pytest.mark.parametrize("focal_length", [5e-3])
@pytest.mark.parametrize("n_medium, NA", [(1.5, 1.49)])
@pytest.mark.parametrize("x_shift, y_shift", [(1.0e-6, -0.9e-6)])
def test_czt_against_summation(focal_length, n_medium, NA, x_shift, y_shift):
    """Test the czt-based implementation against an implementation which sums plane waves. The
    outcomes should be equivalent within machine precision"""

    objective = Objective(NA=NA, focal_length=focal_length, n_medium=n_medium, n_bfp=1.0)
    bfp_sampling_n = 50
    lambda_vac = 1064e-9
    filling_factor = 0.9

    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, 10)
    x_eval = np.linspace(-lambda_vac, lambda_vac, 10) + x_shift
    y_eval = np.linspace(-1.1 * lambda_vac, 1.1 * lambda_vac, 11) + y_shift

    Ex_pw, Ey_pw, Ez_pw = focus_gaussian_quad(
        objective,
        lambda_vac=lambda_vac,
        filling_factor=filling_factor,
        x=x_eval,
        y=y_eval,
        z=z_eval,
        integration_order=bfp_sampling_n,
        return_grid=False,
        integration_method="equidistant",
    )

    Ex_czt, Ey_czt, Ez_czt = focus_gaussian_czt(
        objective=objective,
        lambda_vac=lambda_vac,
        filling_factor=filling_factor,
        x_range=(np.min(x_eval), np.max(x_eval)),
        numpoints_x=10,
        y_range=(np.min(y_eval), np.max(y_eval)),
        numpoints_y=11,
        z=z_eval,
        bfp_sampling_n=bfp_sampling_n,
        return_grid=False,
    )

    np.testing.assert_allclose([Ex_pw, Ey_pw, Ez_pw], [Ex_czt, Ey_czt, Ez_czt])

"""Test the czt-based implementation against an implementation which sums plane waves. The outcomes
should be equivalent within machine precision"""

import numpy as np
import pytest

from lumicks.pyoptics.psf.czt import Objective, focus_gaussian_czt
from lumicks.pyoptics.psf.quad import focus_gaussian_quad


@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2)])
@pytest.mark.parametrize(
    "x_shift, y_shift", [(1.0e-6, -0.9e-6), (-1.23e-6, -0.2e-6), (0.75e-6, 1.65e-6)]
)
def test_focus_gaussian_czt(focal_length, n_medium, NA, x_shift, y_shift):
    objective = Objective(NA=NA, focal_length=focal_length, n_medium=n_medium, n_bfp=1.0)
    bfp_sampling_n = 125
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

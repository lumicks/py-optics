import numpy as np
import pytest
from scipy.constants import (
    speed_of_light as _C,
    epsilon_0 as _EPS0,
)
import lumicks.pyoptics.trapping as trp
import lumicks.pyoptics.psf as psf


@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
def test_gaussian_input(
    focal_length, n_medium, NA
):
    bead_size = 1e-9
    lambda_vac = 1.064e-6
    n_bead = n_medium
    n_bfp = 1.0
    num_pts = 21
    filling_factor = 0.9
    w0 = filling_factor * focal_length * NA / n_medium
    z_eval = 0  # np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    xy_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    bead = trp.Bead(bead_size, n_bead, n_medium, lambda_vac)
    bfp_sampling = 31
    objective = trp.Objective(
        NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

    # Power to get E0 = 1 V/m, to compare with fast_gauss_psf()
    P = 1/4 * 1**2 * _C * n_bfp * np.pi * w0**2 * _EPS0

    Exm, Eym, Ezm = trp.fields_focus_gaussian(
        P, filling_factor, objective, bead, x=xy_eval, y=xy_eval, z=z_eval,
        bead_center=(0, 0, 0), bfp_sampling_n=bfp_sampling,
        return_grid=False, verbose=True, num_orders=None
    )

    Exf, Eyf, Ezf = psf.fast_gauss_psf(
        lambda_vac, n_bfp=n_bfp, n_medium=n_medium, focal_length=focal_length,
        filling_factor=filling_factor, NA=NA,
        xrange=4*lambda_vac, numpoints_x=num_pts, yrange=4*lambda_vac,
        numpoints_y=num_pts, z=z_eval, bfp_sampling_n=bfp_sampling,
        return_grid=False
    )

    np.testing.assert_allclose(
        [Exf, Eyf, Ezf], [Exm, Eym, Ezm], rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    "bead_center", [(1e-6, 0.4e-6, -0.5e-6), (-0.2e-6, -0.33e-6, 0.554e-6)]
)
def test_gaussian_input_bead_shift(bead_center):
    bead_size = 1e-9
    lambda_vac = 1.064e-6
    num_pts = 21
    NA = 1.2
    n_medium = 1.5
    n_bead = n_medium
    focal_length = 4.43e-3
    filling_factor = 0.9
    w0 = filling_factor * focal_length * NA / n_medium
    n_bfp = 1.0
    z_eval = 0  # np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    xy_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    bead = trp.Bead(bead_size, n_bead, n_medium, lambda_vac)
    objective = trp.Objective(
        NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)
    bfp_sampling = 31

    # Power to get E0 = 1 V/m, to compare with fast_gauss_psf()
    P = 1/4 * 1**2 * _C * n_bfp * np.pi * w0**2 * _EPS0

    Exm, Eym, Ezm = trp.fields_focus_gaussian(
        P, filling_factor, objective, bead, x=xy_eval, y=xy_eval, z=z_eval,
        bead_center=(0, 0, 0), bfp_sampling_n=bfp_sampling,
        return_grid=False, verbose=True, num_orders=None
    )
    Exr, Eyr, Ezr = psf.focused_gauss(
        lambda_vac, n_bfp=n_bfp, n_medium=n_medium, focal_length=focal_length,
        filling_factor=filling_factor, NA=NA, x=xy_eval, y=xy_eval, z=z_eval,
        bfp_sampling_n=bfp_sampling, return_grid=False
    )

    Exf, Eyf, Ezf = psf.fast_gauss_psf(
        lambda_vac, n_bfp=n_bfp, n_medium=n_medium, focal_length=focal_length,
        filling_factor=filling_factor, NA=NA, xrange=4*lambda_vac,
        numpoints_x=num_pts, yrange=4*lambda_vac, numpoints_y=num_pts,
        z=z_eval, bfp_sampling_n=bfp_sampling, return_grid=False
    )

    np.testing.assert_allclose(
        [Exm, Eym, Ezm], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        [Exf, Eyf, Ezf], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=[]
import numpy as np
import numpy.testing
import pytest

# %% tags=[]
import pyoptics.mie_calc as mc
import pyoptics.fast_psf_calc as psf


# %%
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
def test_gaussian_input(
    focal_length, n_medium, NA
):
    bead_size = 1e-9
    lambda_vac = 1.064e-6
    n_bead =  n_medium
    num_pts = 21
    z_eval = 0  # np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    xy_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)
    bfp_sampling = 31
    
    Exm, Eym, Ezm, X, Y, Z = mie.fields_gaussian_focus(n_BFP=1.0, focal_length=focal_length, NA=NA, filling_factor=1.0,x=xy_eval, y=xy_eval, z=z_eval, 
                                         bead_center=(0,0,0), bfp_sampling_n=bfp_sampling, return_grid=True, verbose=True, num_orders=None)
    Exr, Eyr, Ezr = psf.focused_gauss(lambda_vac, n_bfp=1.0, n_medium=n_medium, focal_length=focal_length, filling_factor=1.0,
        NA=NA, x=xy_eval, y=xy_eval, z=z_eval, bfp_sampling_n=bfp_sampling, return_grid=False)
    
    Exf, Eyf, Ezf = psf.fast_gauss_psf(lambda_vac, n_bfp=1.0, n_medium=n_medium, focal_length=focal_length, filling_factor=1.0,
        NA=NA, xrange=4*lambda_vac, numpoints_x=num_pts, yrange=4*lambda_vac, numpoints_y=num_pts, z=z_eval, bfp_sampling_n=bfp_sampling, return_grid=False)
    
    #return [Exm, Eym, Ezm, Exr, Eyr, Ezr, Exf, Eyf, Ezf]
    np.testing.assert_allclose([Exm, Eym, Ezm], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose([Exf, Eyf, Ezf], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)



# %%
@pytest.mark.parametrize("bead_center", [(1e-6, 0.4e-6, -0.5e-6),(-0.2e-6, -0.33e-6, 0.554e-6)])
def test_gaussian_input_bead_shift(bead_center):
    bead_size = 1e-9
    lambda_vac = 1.064e-6
    n_bead =  1.5
    num_pts = 21
    NA = 1.2
    n_medium=1.5
    focal_length = 4.43e-3
    z_eval = 0  # np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    xy_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)
    bfp_sampling = 31
    
    Exm, Eym, Ezm, X, Y, Z = mie.fields_gaussian_focus(n_BFP=1.0, focal_length=focal_length, NA=NA, filling_factor=1.0,x=xy_eval, y=xy_eval, z=z_eval, 
                                         bead_center=(0,0,0), bfp_sampling_n=bfp_sampling, return_grid=True, verbose=True, num_orders=None)
    Exr, Eyr, Ezr = psf.focused_gauss(lambda_vac, n_bfp=1.0, n_medium=n_medium, focal_length=focal_length, filling_factor=1.0,
        NA=NA, x=xy_eval, y=xy_eval, z=z_eval, bfp_sampling_n=bfp_sampling, return_grid=False)
    
    Exf, Eyf, Ezf = psf.fast_gauss_psf(lambda_vac, n_bfp=1.0, n_medium=n_medium, focal_length=focal_length, filling_factor=1.0,
        NA=NA, xrange=4*lambda_vac, numpoints_x=num_pts, yrange=4*lambda_vac, numpoints_y=num_pts, z=z_eval, bfp_sampling_n=bfp_sampling, return_grid=False)
    
    #return [Exm, Eym, Ezm, Exr, Eyr, Ezr, Exf, Eyf, Ezf]
    np.testing.assert_allclose([Exm, Eym, Ezm], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose([Exf, Eyf, Ezf], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)

# %%

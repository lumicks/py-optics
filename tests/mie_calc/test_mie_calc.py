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
    num_pts = 11
    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    xy_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)
    
    Exm, Eym, Ezm, X, Y, Z = mie.fields_gaussian_focus(n_BFP=1.0, focal_length=focal_length, NA=NA, filling_factor=1.0,x=xy_eval, y=xy_eval, z=z_eval, 
                                         bead_center=(0,0,0), bfp_sampling_n=31, return_grid=True, verbose=True, num_orders=1)
    Exr, Eyr, Ezr = psf.fast_gauss_psf(lambda_vac, n_bfp=1.0, n_medium=n_medium, focal_length=focal_length, filling_factor=1.0,
        NA=NA, xrange=4*lambda_vac, numpoints_x=num_pts, yrange=4*lambda_vac, numpoints_y=num_pts, z=z_eval, bfp_sampling_n=31, return_grid=False)
    
    #return [Exm, Eym, Ezm, Exr, Eyr, Ezr]
    np.testing.assert_allclose([Exm, Eym, Ezm], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)
    

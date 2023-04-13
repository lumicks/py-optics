# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=[]
import numpy as np
import pytest
import matplotlib.pyplot as plt

# %% tags=[]
import pyoptics.mie_calc as mc
import pyoptics.fast_psf_calc as psf
from scipy.constants import (
    speed_of_light as _C,
    epsilon_0 as _EPS0,
)


# %%
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
def test_gaussian_input(
    focal_length, n_medium, NA
):
    bead_size = 1e-9
    lambda_vac = 1.064e-6
    n_bead =  n_medium
    n_bfp = 1.0
    num_pts = 51
    filling_factor = 0.9
    w0 = filling_factor * focal_length * NA / n_medium
    z_eval = 0  # np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    xy_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    bead = mc.Bead(bead_size, n_bead, n_medium, lambda_vac)
    bfp_sampling = 31
    objective = mc.Objective(NA=1.2, focal_length=4.43e-3, n_bfp=1.0, n_medium=1.33)
    # P = 1/4 * 1**2 * _C * n_bfp * np.pi * w0**2 * _EPS0
    # print(P)
    
    def gaussian_beam(X_bfp, Y_bfp, **kwargs): 
        Ex = np.exp(-(X_bfp**2 + Y_bfp**2) / w0**2)
        return (Ex, None)

    Exm, Eym, Ezm = mc.fields_focus(gaussian_beam, objective, bead, x=xy_eval, y=xy_eval, z=z_eval,
        bead_center=(0,0,0), bfp_sampling_n=bfp_sampling,
        return_grid=False, verbose=True, num_orders=None
    )
    # Exr, Eyr, Ezr = psf.focused_gauss(
    #     lambda_vac, n_bfp=1.0, n_medium=n_medium,
    #     focal_length=focal_length, filling_factor=1.0,
    #     NA=NA, x=xy_eval, y=xy_eval, z=z_eval,
    #     bfp_sampling_n=bfp_sampling, return_grid=False
    # )
    
    Exf, Eyf, Ezf = psf.fast_gauss_psf(lambda_vac, n_bfp=1.0, n_medium=n_medium, focal_length=focal_length, filling_factor=filling_factor,
        NA=NA, xrange=4*lambda_vac, numpoints_x=num_pts, yrange=4*lambda_vac, numpoints_y=num_pts, z=z_eval, bfp_sampling_n=bfp_sampling, return_grid=False)
    
    plt.figure()
    plt.imshow(np.imag(Exm))
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.imshow(np.imag(Exf))
    plt.colorbar()
    plt.show()
    #return [Exm, Eym, Ezm, Exr, Eyr, Ezr, Exf, Eyf, Ezf]
    np.testing.assert_allclose([Exm, Eym, Ezm], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)
    # np.testing.assert_allclose([Exf, Eyf, Ezf], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)



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
    
    Exm, Eym, Ezm, X, Y, Z = mie.fields_gaussian_focus(n_bfp=1.0, focal_length=focal_length, NA=NA, filling_factor=1.0,x=xy_eval, y=xy_eval, z=z_eval, 
                                         bead_center=(0,0,0), bfp_sampling_n=bfp_sampling, return_grid=True, verbose=True, num_orders=None)
    Exr, Eyr, Ezr = psf.focused_gauss(lambda_vac, n_bfp=1.0, n_medium=n_medium, focal_length=focal_length, filling_factor=1.0,
        NA=NA, x=xy_eval, y=xy_eval, z=z_eval, bfp_sampling_n=bfp_sampling, return_grid=False)
    
    Exf, Eyf, Ezf = psf.fast_gauss_psf(lambda_vac, n_bfp=1.0, n_medium=n_medium, focal_length=focal_length, filling_factor=1.0,
        NA=NA, xrange=4*lambda_vac, numpoints_x=num_pts, yrange=4*lambda_vac, numpoints_y=num_pts, z=z_eval, bfp_sampling_n=bfp_sampling, return_grid=False)
    
    #return [Exm, Eym, Ezm, Exr, Eyr, Ezr, Exf, Eyf, Ezf]
    np.testing.assert_allclose([Exm, Eym, Ezm], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose([Exf, Eyf, Ezf], [Exr, Eyr, Ezr], rtol=1e-8, atol=1e-8)


# %%
test_gaussian_input(4.43e-3, 1.33, 1.2)

# %%
import logging

# %% tags=[]
logging.getLogger().setLevel('DEBUG')

# %%

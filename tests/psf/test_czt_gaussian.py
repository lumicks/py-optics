import numpy as np
import pytest
import lumicks.pyoptics.psf as psf


@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
def test_fast_gaussian(
    focal_length, n_medium, NA
):
    n_bfp=1.0
    bfp_sampling_n=125
    lambda_vac=1064e-9
    filling_factor=0.9
    
    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, 10)
    xy_eval = np.linspace(-lambda_vac, lambda_vac, 10)
    
    Ex_pw, Ey_pw, Ez_pw = psf.focused_gauss(lambda_vac=lambda_vac, n_bfp=n_bfp, n_medium=n_medium, 
                                              focal_length=focal_length, NA=NA, filling_factor=filling_factor, 
                                              x=xy_eval, y=xy_eval, z=z_eval, 
                                              bfp_sampling_n=bfp_sampling_n, return_grid=False)
    
    Ex_czt, Ey_czt, Ez_czt = psf.fast_gauss_psf(lambda_vac=lambda_vac, n_bfp=n_bfp, n_medium=n_medium, 
                                              focal_length=focal_length, NA=NA, filling_factor=filling_factor, 
                                              xrange=lambda_vac * 2, numpoints_x=10, yrange=lambda_vac * 2, 
                                              numpoints_y=10, z=z_eval, 
                                              bfp_sampling_n=bfp_sampling_n, return_grid=False)
    
    np.testing.assert_allclose([Ex_pw, Ey_pw, Ez_pw], [Ex_czt, Ey_czt, Ez_czt])


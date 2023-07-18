import numpy as np
import pytest
from lumicks.pyoptics.psf.reference import focused_gauss_ref
from lumicks.pyoptics.psf.direct import focused_gauss

@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
def test_gaussian(
    focal_length, n_medium, NA
):
    n_bfp=1.0
    bfp_sampling_n=125
    lambda_vac=1064e-9
    filling_factor=0.9
    
    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, 10)
    xy_eval = np.linspace(-lambda_vac, lambda_vac, 10)
    
    Ex_ref, Ey_ref, Ez_ref = focused_gauss_ref(lambda_vac=lambda_vac, n_bfp=n_bfp, n_medium=n_medium, 
                                                   focal_length=focal_length, filling_factor=filling_factor, NA=NA, 
                                                   x=xy_eval, y=xy_eval,z=z_eval)
    
    Ex, Ey, Ez = focused_gauss(lambda_vac=lambda_vac, n_bfp=n_bfp, n_medium=n_medium, 
                                              focal_length=focal_length, NA=NA, filling_factor=filling_factor, 
                                              x=xy_eval, y=xy_eval, z=z_eval, 
                                              bfp_sampling_n=bfp_sampling_n, return_grid=False)
    # Allow 1 V/m absolute error and 5% relative error
    np.testing.assert_allclose([Ex_ref, Ey_ref, Ez_ref], [Ex, Ey, Ez], rtol=.05, atol=1)


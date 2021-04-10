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
#     display_name: Python 3.8 (XPython)
#     language: python
#     name: xpython
# ---

# %%
# %matplotlib inline
import numpy as np
import numpy.testing
import pytest

# %% tags=[]
import pyoptics.fast_psf_calc as psf
skip = False
theta = 0
phi = 0


# %%
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
def test_plane_wave(
    focal_length, n_medium, NA, n_bfp=1.0, bfp_sampling_n=7, lambda_vac=1064e-9
):
    global skip
    global theta
    global phi
    
    k = 2*np.pi*n_medium / lambda_vac
    ks = k * NA / n_medium
    dk = ks / (bfp_sampling_n - 1)
    
    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, 10)
    xy_eval = np.linspace(-lambda_vac, lambda_vac, 10)
    
    M = 2 * bfp_sampling_n - 1
    for p in range(M):
        for m in range(M):
            theta = 0
            phi = 0
            skip = False
            
            def input_field_Etheta(X_BFP, Y_BFP, R, Rmax, Theta, Phi):
                #Create an input field that is theta-polarized with 1 V/m after refraction by the lens and propagation to the focal plane
                global skip
                global theta
                global phi
                Ex = np.zeros(X_BFP.shape, dtype='complex128')
                Ey = np.zeros(X_BFP.shape, dtype='complex128')
                correction = k*np.cos(Theta[p,m]) *(n_medium/n_bfp)**0.5 * np.cos(Theta[p,m])**-0.5
                Expoint = np.cos(Phi[p,m]) 
                Eypoint = np.sin(Phi[p,m]) 
                Ex[p, m] = Expoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                Ey[p, m] = Eypoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                theta = Theta[p, m]
                phi = Phi[p, m]
                if R[p, m] >  Rmax:
                    skip = True
                return (Ex, Ey)
            
            def input_field_Ephi(X_BFP, Y_BFP, R, Rmax, Theta, Phi):
                #Create an input field that is phi-polarized with 1 V/m after refraction by the lens and propagation to the focal plane
                global skip 
                global theta
                global phi
                Ex = np.zeros(X_BFP.shape, dtype='complex128')
                Ey = np.zeros(X_BFP.shape, dtype='complex128')
                correction = k*np.cos(Theta[p,m]) *(n_medium/n_bfp)**0.5 * np.cos(Theta[p,m])**-0.5
                Expoint = -np.sin(Phi[p,m]) 
                Eypoint = np.cos(Phi[p,m]) 
                Ex[p, m] = Expoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                Ey[p, m] = Eypoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                theta = Theta[p, m]
                phi = Phi[p, m]
                if R[p, m] >  Rmax:
                    skip = True
                return (Ex, Ey)

            Ex, Ey, Ez, X, Y, Z = psf.direct_psf_calc(input_field_Etheta, lambda_vac=lambda_vac, n_bfp=n_bfp, n_medium=n_medium, 
                                                      focal_length=focal_length, NA=NA, x=xy_eval, y=xy_eval, z=z_eval, 
                                                      bfp_sampling_n=bfp_sampling_n, return_grid=True)
            if skip:
                continue

            kz = k * np.cos(theta)
            kx = - k * np.sin(theta) * np.cos(phi)
            ky = - k * np.sin(theta) * np.sin(phi)
            
            # Check convention, +1j for k vector as we use -1j for time phasor
            Exp = np.exp(1j * (kx * X + ky * Y + kz * z_eval))
            Expw = np.cos(theta) * np.cos(phi) * Exp
            Eypw = np.cos(theta) * np.sin(phi) * Exp
            Ezpw = np.sin(theta) * Exp
            
            np.testing.assert_allclose([Ex, Ey,Ez], [Expw, Eypw, Ezpw])
            np.testing.assert_allclose(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2, np.ones(Ex.shape))
            
            Ex, Ey, Ez, X, Y, Z = psf.direct_psf_calc(input_field_Ephi, lambda_vac=lambda_vac, n_bfp=n_bfp, n_medium=n_medium, 
                                                      focal_length=focal_length, NA=NA, x=xy_eval, y=xy_eval, z=z_eval, 
                                                      bfp_sampling_n=bfp_sampling_n, return_grid=True)
            
            Expw = -np.sin(phi) * Exp
            Eypw = np.cos(phi) * Exp
            Ezpw = np.zeros(Ex.shape) * Exp
            # Ezpw == 0, but Ez has some numerical rounding errors. Therefore specify an absolute tolerance that is acceptable
            np.testing.assert_allclose([Ex, Ey,Ez], [Expw, Eypw, Ezpw], atol=1e-14)
            np.testing.assert_allclose(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2, np.ones(Ex.shape))


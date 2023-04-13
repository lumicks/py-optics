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
import pyoptics.mie_calc as mc


# %%
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("n_angles", [7, 22])
def test_plane_wave_direct(
    n_medium, NA, n_angles, lambda_vac=1064e-9
):
    k = 2*np.pi*n_medium / lambda_vac
    
    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, 21)
    xy_eval = np.linspace(-lambda_vac, lambda_vac, 21)
    
    sin_th_max = NA / n_medium
    sin_th = np.linspace(-sin_th_max, sin_th_max, num=n_angles)
    X_BFP, Y_BFP = np.meshgrid(sin_th, sin_th)
    R_BFP = np.hypot(X_BFP, Y_BFP)
    aperture = R_BFP < sin_th_max
    theta = np.zeros(R_BFP.shape)
    theta[aperture] = np.arcsin(R_BFP[aperture])
    
    phi = np.arctan2(Y_BFP, X_BFP)
    phi[np.logical_not(aperture)] = 0
    phi[R_BFP == 0] = 0
    
    for p in range(n_angles):
        for m in range(n_angles):
            if not aperture[p,m]:
                continue
            bead = mc.Bead(1e-9, n_medium, n_medium, lambda_vac)
            Ex, Ey, Ez, X, Y, Z = mc.fields_plane_wave(bead, x=xy_eval, y=xy_eval, z=z_eval, theta=theta[p,m], phi=phi[p,m],
                                                        polarization=(1,0), return_grid=True, verbose=False)
            kz = k * np.cos(theta[p,m])
            kx = - k * np.sin(theta[p,m]) * np.cos(phi[p,m])
            ky = - k * np.sin(theta[p,m]) * np.sin(phi[p,m])
            
            # Check convention, +1j for k vector as we use -1j for time phasor
            Exp = np.exp(1j * (kx * X + ky * Y + kz * z_eval))
            Expw = np.cos(theta[p,m]) * np.cos(phi[p,m]) * Exp
            Eypw = np.cos(theta[p,m]) * np.sin(phi[p,m]) * Exp
            Ezpw = np.sin(theta[p,m]) * Exp
            
            #return [Expw, Eypw, Ezpw, Ex, Ey, Ez]
            np.testing.assert_allclose(Ex, Expw, atol=1e-14)
            np.testing.assert_allclose(Ey, Eypw, atol=1e-14)
            np.testing.assert_allclose(Ez, Ezpw, atol=1e-14)
            np.testing.assert_allclose(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2, np.ones(Ex.shape))
            
            Ex, Ey, Ez, X, Y, Z = mc.fields_plane_wave(bead, x=xy_eval, y=xy_eval, z=z_eval, theta=theta[p,m], phi=phi[p,m],
                                                        polarization=(0, 1), return_grid=True, verbose=False)
            Expw = -np.sin(phi[p,m]) * Exp
            Eypw = np.cos(phi[p,m]) * Exp
            Ezpw = np.zeros(Ex.shape) * Exp
            
            
            # Ezpw == 0, but Ez has some numerical rounding errors. Therefore specify an absolute tolerance that is acceptable
            np.testing.assert_allclose(Ex, Expw, atol=1e-14)
            np.testing.assert_allclose(Ey, Eypw, atol=1e-14)
            np.testing.assert_allclose(Ez, Ezpw, atol=1e-14)
            np.testing.assert_allclose(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2, np.ones(Ex.shape))


# %%
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
def test_plane_wave_bfp(
    focal_length, n_medium, NA, n_bfp=1.0, bfp_sampling_n=7, lambda_vac=1064e-9
):
    global skip
    global costheta
    global cosphi
    global sinphi
    
    num_pts = 21
    
    k = 2*np.pi*n_medium / lambda_vac
    ks = k * NA / n_medium
    dk = ks / (bfp_sampling_n - 1)
    
    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    xy_eval = np.linspace(-lambda_vac, lambda_vac, num_pts)
    
    M = 2 * bfp_sampling_n - 1
    for p in range(M):
        for m in range(M):
            costheta = 0
            cosphi = 0
            sinphi = 0
            skip = False
            
            def input_field_Etheta(X_BFP, Y_BFP, R, Rmax, cosTheta, cosPhi, sinPhi):
                #Create an input field that is theta-polarized with 1 V/m after refraction by the lens and propagation to the focal plane
                global skip
                global costheta
                global cosphi
                global sinphi
                Ex = np.zeros(X_BFP.shape, dtype='complex128')
                Ey = np.zeros(X_BFP.shape, dtype='complex128')
                if R[p, m] >  Rmax:
                    skip = True
                    return (Ex, Ey)

                correction = k*cosTheta[p,m] *(n_medium/n_bfp)**0.5 * cosTheta[p,m]**-0.5
                Expoint = cosPhi[p,m]
                Eypoint = sinPhi[p,m]

                Ex[p, m] = Expoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                Ey[p, m] = Eypoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                costheta = cosTheta[p, m]
                cosphi = cosPhi[p, m]
                sinphi = sinPhi[p, m]
                print(p,m)
                return (Ex, Ey)
            
            def input_field_Ephi(X_BFP, Y_BFP, R, Rmax, cosTheta, cosPhi, sinPhi):
                #Create an input field that is phi-polarized with 1 V/m after refraction by the lens and propagation to the focal plane
                global skip 
                global costheta
                global cosphi
                global sinphi
                Ex = np.zeros(X_BFP.shape, dtype='complex128')
                Ey = np.zeros(X_BFP.shape, dtype='complex128')
                if R[p, m] >  Rmax:
                    skip = True
                correction = k*cosTheta[p,m] *(n_medium/n_bfp)**0.5 * cosTheta[p,m]**-0.5
                Expoint = -sinPhi[p,m]
                Eypoint = cosPhi[p,m] 
                Ex[p, m] = Expoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                Ey[p, m] = Eypoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                costheta = cosTheta[p, m]
                cosphi = cosPhi[p, m]
                sinphi = sinPhi[p,m]
                
                return (Ex, Ey)
            
            bead = mc.Bead(1e-9, n_medium, n_medium, lambda_vac)
            objective = mc.Objective(n_bfp=n_bfp, focal_length=focal_length, NA=NA, n_medium=bead.n_medium)
            Ex, Ey, Ez, X, Y, Z = mc.fields_focus(input_field_Etheta, bead=bead, objective=objective, 
                                                      x=xy_eval, y=0, z=z_eval, 
                                                      bfp_sampling_n=bfp_sampling_n, return_grid=True, verbose=False)
            if skip:
                continue

            kz = k * costheta
            kx = - k * (1-costheta**2)**0.5 * cosphi
            ky = - k * (1-costheta**2)**0.5 * sinphi
            
            # Check convention, +1j for k vector as we use -1j for time phasor
            Exp = np.exp(1j * (kx * X + ky * Y + kz * Z))
            Expw = costheta * cosphi * Exp
            Eypw = costheta * sinphi * Exp
            Ezpw = (1-costheta**2)**0.5 * Exp
            
            np.testing.assert_allclose(Ex, Expw, atol=1e-14)
            np.testing.assert_allclose(Ey, Eypw, atol=1e-14)
            np.testing.assert_allclose(Ez, Ezpw, atol=1e-14)
            np.testing.assert_allclose(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2, np.ones(Ex.shape))
#             finally:
#                 print(costheta, cosphi, sinphi)
#                 return Expw, Eypw, Ezpw, Ex,Ey,Ez, X, Y, Z, Exp
            
            Ex, Ey, Ez, X, Y, Z = mc.fields_focus(input_field_Ephi, bead=bead, objective=objective, 
                                                      x=xy_eval, y=0, z=z_eval, 
                                                      bfp_sampling_n=bfp_sampling_n, return_grid=True, verbose=False)
            Expw = -sinphi * Exp
            Eypw = cosphi * Exp
            Ezpw = np.zeros(Ex.shape)
            # Ezpw == 0, but Ez has some numerical rounding errors. Therefore specify an absolute tolerance that is acceptable
            np.testing.assert_allclose(Ex, Expw, atol=1e-14)
            np.testing.assert_allclose(Ey, Eypw, atol=1e-14)
            np.testing.assert_allclose(Ez, Ezpw, atol=1e-14)
            np.testing.assert_allclose(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2, np.ones(Ex.shape))


# %%

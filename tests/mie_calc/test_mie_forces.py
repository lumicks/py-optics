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

# %% tags=[]
import numpy as np
import pytest
from pyoptics import mie_calc as mc

# %% tags=[]
_C = 299792458
_MU0 = 4 * np.pi * 1e-7
_EPS0 = (_C**2 * _MU0)**-1


# %%
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
def test_plane_wave_forces_bfp(
    focal_length, n_medium, NA, n_bfp=1.0, bfp_sampling_n=7, lambda_vac=1064e-9
):
    global skip
    global costheta
    global cosphi
    global sinphi
    
    n_bead = 2.1
    bead_size = 500e-9  # larger than dipole approximation is valid for
    E0 = 2.2
    mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)
    Fpr = mie.pressure_eff() * np.pi * mie.bead_diameter**2 / 8 * E0**2 * mie.n_medium**2 * _EPS0
    k = 2*np.pi*n_medium / lambda_vac
    ks = k * NA / n_medium
    dk = ks / (bfp_sampling_n - 1)
    
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
                Expoint = cosPhi[p,m]*E0
                Eypoint = sinPhi[p,m]*E0

                Ex[p, m] = Expoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                Ey[p, m] = Eypoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                costheta = cosTheta[p, m]
                cosphi = cosPhi[p, m]
                sinphi = sinPhi[p, m]
                
                return (Ex, Ey)
            
            def input_field_Ephi(X_BFP, Y_BFP, R, Rmax, cosTheta, cosPhi, sinPhi):
                #Create an input field that is phi-polarized with 1 V/m after refraction by the lens and propagation to the focal plane
                global skip 
                global theta
                global phi
                Ex = np.zeros(X_BFP.shape, dtype='complex128')
                Ey = np.zeros(X_BFP.shape, dtype='complex128')
                if R[p, m] >  Rmax:
                    skip = True
                
                correction = k*cosTheta[p,m] *(n_medium/n_bfp)**0.5 * cosTheta[p,m]**-0.5
                Expoint = -sinPhi[p,m]*E0
                Eypoint = cosPhi[p,m]*E0
                Ex[p, m] = Expoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                Ey[p, m] = Eypoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                costheta = cosTheta[p, m]
                cosphi = cosPhi[p, m]
                sinphi = sinPhi[p, m]
                
                return (Ex, Ey)
            
            F = mie.forces_focused_fields(input_field_Etheta, n_bfp=n_bfp, focal_length=focal_length, NA=NA, 
                                         bead_center=(0,0,0), bfp_sampling_n=bfp_sampling_n, verbose=False, num_orders=13)
            if skip:
                continue

            # direction of the plane wave, hence direction of the force
            nz = costheta
            nx = - (1 - costheta**2)**0.5 * cosphi
            ny = - (1 - costheta**2)**0.5 * sinphi
            n = [nx, ny, nz]
            Fn = np.squeeze(F/np.linalg.norm(F))
            
            # check that the magnitude is the same as predicted for Mie scattering
            np.testing.assert_allclose(Fpr, np.linalg.norm(F), rtol=1e-2)
            #check that the force direction is in the same direction as the plane wave
            np.testing.assert_allclose(n, Fn, rtol=1e-2, atol=1e-4)
            
            
            F = mie.forces_focused_fields(input_field_Ephi, n_bfp=n_bfp, focal_length=focal_length, NA=NA, 
                                         bead_center=(0,0,0), bfp_sampling_n=bfp_sampling_n, verbose=False, num_orders=13)
            Fn = np.squeeze(F/np.linalg.norm(F))
            # check that the magnitude is the same as predicted for Mie scattering
            np.testing.assert_allclose(Fpr, np.linalg.norm(F), rtol=1e-2, atol=1e-4)
            #check that the force direction is in the same direction as the plane wave
            np.testing.assert_allclose(n, Fn, rtol=1e-2, atol=1e-4)


# %%
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
def test_plane_wave_dipole_forces_bfp(
    focal_length, n_medium, NA, n_bfp=1.0, bfp_sampling_n=7, lambda_vac=1064e-9
):
    global skip
    global costheta
    global cosphi
    global sinphi
    
    rng = np.random.default_rng()

    n_bead = 5
    bead_size = 20e-9
    E0 = 10.9
    
    k = 2*np.pi*n_medium / lambda_vac
    ks = k * NA / n_medium
    dk = ks / (bfp_sampling_n - 1)
    
    M = 2 * bfp_sampling_n - 1
    mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)
    # quasi-static polarizability
    a_s = 4 * np.pi * _EPS0 * n_medium**2 * (bead_size/2)**3 * (n_bead**2 - n_medium**2)/(n_bead**2 + 2 * n_medium**2)

    # correct for radiation reaction
    a = a_s + 1j * k**3 / (6*np.pi*_EPS0*n_medium**2) * a_s**2

    # also get polarizability directly from Mie coefficient a1
    mie._get_mie_coefficients()
    print(mie._an.shape)
    a1 = mie._an[0]
    alpha = _EPS0 * n_medium**2 * 6j * np.pi * a1 / k**3

    # Use dipole approximation and Eq. 14.40 from "principles of nano-optics, L Novotny & B. Hecht, 2nd ed."
    # Ei grad(Ei) for an x-polarized plane wave in the z direction
    Ex_gradEx = E0**2 * 1j * k  # in z direction only
    Fdipole_quasistatic = a.real / 2 * Ex_gradEx.real + a.imag / 2 * Ex_gradEx.imag
    Fdipole_mie = alpha.real / 2 * Ex_gradEx.real + alpha.imag / 2 * Ex_gradEx.imag

    for p in range(M):
        for m in range(M):
            costheta = 0
            cosphi = 0
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
                Expoint = cosPhi[p,m]*E0
                Eypoint = sinPhi[p,m]*E0

                Ex[p, m] = Expoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                Ey[p, m] = Eypoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                costheta = cosTheta[p, m]
                cosphi = cosPhi[p, m]
                sinphi = sinPhi[p, m]
                
                return (Ex, Ey)
            
            def input_field_Ephi(X_BFP, Y_BFP, R, Rmax, cosTheta, cosPhi, sinPhi):
                global skip 
                global theta
                global phi
                Ex = np.zeros(X_BFP.shape, dtype='complex128')
                Ey = np.zeros(X_BFP.shape, dtype='complex128')
                if R[p, m] >  Rmax:
                    skip = True
                
                correction = k*cosTheta[p,m] *(n_medium/n_bfp)**0.5 * cosTheta[p,m]**-0.5
                Expoint = -sinPhi[p,m]*E0
                Eypoint = cosPhi[p,m]*E0
                Ex[p, m] = Expoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                Ey[p, m] = Eypoint * correction * 2 * np.pi / (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                costheta = cosTheta[p, m]
                cosphi = cosPhi[p, m]
                sinphi = sinPhi[p, m]
                
                return (Ex, Ey)
            
            bead_pos = np.squeeze(rng.random((3,1)))*200e-9
            F = mie.forces_focused_fields(input_field_Etheta, n_bfp=n_bfp, focal_length=focal_length, NA=NA, 
                                         bead_center=bead_pos, bfp_sampling_n=bfp_sampling_n, verbose=False, num_orders=5)
            if skip:
                continue
            
            # direction of the plane wave, hence direction of the force
            nz = costheta
            nx = - (1 - costheta**2)**0.5 * cosphi
            ny = - (1 - costheta**2)**0.5 * sinphi
            n = [nx, ny, nz]
            Fn = np.squeeze(F/np.linalg.norm(F))
            Fpr = mie.pressure_eff() * np.pi * mie.bead_diameter**2 / 4 * 0.5 *E0**2 * mie.n_medium**2 * _EPS0
            
            # check that the magnitude is the same as predicted for Mie scattering
            np.testing.assert_allclose(Fpr, np.linalg.norm(F), rtol=1e-2, atol=1e-14)
            # check that the force direction is in the same direction as the plane wave
            np.testing.assert_allclose(n, Fn, rtol=1e-2, atol=1e-4)
            # Check that the force is very close to the force obtained through the dipole approximation
            np.testing.assert_allclose(np.linalg.norm(F), Fdipole_quasistatic, rtol=1e-2)
            np.testing.assert_allclose(np.linalg.norm(F), Fdipole_mie, rtol=1e-2)
            
            
            F = mie.forces_focused_fields(input_field_Ephi, n_bfp=n_bfp, focal_length=focal_length, NA=NA, 
                                        bead_center=bead_pos, bfp_sampling_n=bfp_sampling_n, verbose=False, num_orders=5)
            Fn = np.squeeze(F/np.linalg.norm(F))
            # check that the magnitude is the same as predicted for Mie scattering
            np.testing.assert_allclose(Fpr, np.linalg.norm(F), rtol=1e-2, atol=1e-14)
            # check that the force direction is in the same direction as the plane wave
            np.testing.assert_allclose(n, Fn, rtol=1e-2, atol=1e-4)
            # Check that the force is very close to the force obtained through the dipole approximation
            np.testing.assert_allclose(np.linalg.norm(F), Fdipole_quasistatic, rtol=1e-2)
            np.testing.assert_allclose(np.linalg.norm(F), Fdipole_mie, rtol=1e-2)


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

# %%
import numpy as np
import matplotlib.pyplot as plt
import pytest

# %% tags=[]
from pyoptics import fast_psf_calc as psf
from pyoptics import mie_calc as mc

# %% tags=[]
_C = 299792458  # Speed of light [m/s]
_MU0 = 4 * np.pi * 1e-7  # Magnetic permeability
_EPS0 = (_C**2 * _MU0)**-1  # Electric permittivity
n_medium = 1.33
n_bead = 5
bead_size = 20e-9 # [m]
k = 2*np.pi*n_medium / 1064e-9

numpoints = 11
dim = 2e-6
zrange = np.linspace(-dim/2,dim/2,numpoints)
filling_factor = 0.9
NA = 1.2
focal_length = 4.43e-3
bfp_sampling_n = 11
w0 = filling_factor * focal_length * NA / n_medium

# quasi-static polarizability
a_s = 4 * np.pi * _EPS0 * n_medium**2 * (bead_size/2)**3 * (n_bead**2 - n_medium**2)/(n_bead**2 + 2 * n_medium**2)

# correct for radiation reaction
a = a_s + 1j * k**3 / (6*np.pi*_EPS0*n_medium**2) * a_s**2


# %%
def field_func_mie(X_BFP, Y_BFP, *rest):
    
    Ein = np.exp(-((X_BFP)**2 + Y_BFP**2)/w0**2)
    return (Ein, None)

def field_func(X_BFP, Y_BFP, R, Rmax, Th, Phi):
    
    Ein = np.exp(-((X_BFP)**2 + Y_BFP**2)/w0**2)
    return (Ein, None)

def field_func_kx(X_BFP, Y_BFP, R, Rmax, Th, Phi):
    # Takes the derivative of the fields to X
    k = 2*np.pi*n_medium / 1064e-9
    Kz = k * np.cos(Th)
    Kp = k * np.sin(Th)
    Kx = -Kp * np.cos(Phi)
 
    Ein = np.exp(-((X_BFP)**2 + Y_BFP**2)/w0**2)*1j*Kx
    return (Ein, None)


def field_func_ky(X_BFP, Y_BFP, R, Rmax, Th, Phi):
    # Takes the derivative of the fields to Y
    k = 2*np.pi*n_medium / 1064e-9
    Kz = k * np.cos(Th)
    Kp = k * np.sin(Th)
    Ky = -Kp * np.sin(Phi)
    
    Ein = np.exp(-((X_BFP)**2 + Y_BFP**2)/w0**2)*1j*Ky
    return (Ein, None)

def field_func_kz(X_BFP, Y_BFP, R, Rmax, Th, Phi):
    # Takes the derivative of the fields to Z
    k = 2*np.pi*n_medium / 1064e-9
    Kz = k * np.cos(Th)
    
    Ein = np.exp(-((X_BFP)**2 + Y_BFP**2)/w0**2)*1j*Kz
    return (Ein, None)

@pytest.mark.parametrize('z_pos', zrange)
def test_force_focus(z_pos):
    Ex, Ey, Ez, X, Y, Z = psf.fast_psf_calc(field_func, 1064e-9, 1.0, n_medium, 4.43e-3, 1.2, xrange=dim, numpoints_x=numpoints, 
                                               yrange=dim, numpoints_y=numpoints, z=z_pos, bfp_sampling_n=bfp_sampling_n, return_grid=True)
    Exdx, Eydx, Ezdx = psf.fast_psf_calc(field_func_kx, 1064e-9, 1.0, n_medium, 4.43e-3, 1.2, xrange=dim, numpoints_x=numpoints, 
                                               yrange=dim, numpoints_y=numpoints, z=z_pos, bfp_sampling_n=bfp_sampling_n, return_grid=False)
    Exdy, Eydy, Ezdy = psf.fast_psf_calc(field_func_ky, 1064e-9, 1.0, n_medium, 4.43e-3, 1.2, xrange=dim, numpoints_x=numpoints, 
                                               yrange=dim, numpoints_y=numpoints, z=z_pos, bfp_sampling_n=bfp_sampling_n, return_grid=False)
    Exdz, Eydz, Ezdz = psf.fast_psf_calc(field_func_kz, 1064e-9, 1.0, n_medium, 4.43e-3, 1.2, xrange=dim, numpoints_x=numpoints, 
                                               yrange=dim, numpoints_y=numpoints, z=z_pos, bfp_sampling_n=bfp_sampling_n, return_grid=False)

    E_grad_E_x = np.conj(Ex)*Exdx + np.conj(Ey)*Eydx + np.conj(Ez)*Ezdx
    E_grad_E_y = np.conj(Ex)*Exdy + np.conj(Ey)*Eydy + np.conj(Ez)*Ezdy
    E_grad_E_z = np.conj(Ex)*Exdz + np.conj(Ey)*Eydz + np.conj(Ez)*Ezdz
    Fx = np.real(a)/2*E_grad_E_x.real + np.imag(a)/2*E_grad_E_x.imag
    Fy = np.real(a)/2*E_grad_E_y.real + np.imag(a)/2*E_grad_E_y.imag
    Fz = np.real(a)/2*E_grad_E_z.real + np.imag(a)/2*E_grad_E_z.imag
    
    Fx_mie = np.empty(Ex.shape)
    Fy_mie = np.empty(Ex.shape)
    Fz_mie = np.empty(Ex.shape)
    mie = mc.MieCalc(bead_size, n_bead, n_medium, 1064e-9)
    for p in range(numpoints):
        for m in range(numpoints):
            F = mie.forces_focused_fields(field_func_mie, 1.0, 4.43e-3, 1.2, bead_center=(X[p, m], Y[p, m], z_pos), 
                                          bfp_sampling_n=bfp_sampling_n, num_orders=None, integration_orders=None)
            Fx_mie[p,m] = F[0]
            Fy_mie[p,m] = F[1]
            Fz_mie[p,m] = F[2]
    np.testing.assert_allclose(Fx, Fx_mie, rtol=1e-2, atol=1e-23)
    np.testing.assert_allclose(Fy, Fy_mie, rtol=1e-2, atol=1e-23)
    np.testing.assert_allclose(Fz, Fz_mie, rtol=1e-2, atol=1e-23)



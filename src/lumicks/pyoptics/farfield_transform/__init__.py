"""
References:
    1. Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
       Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193

This implementation is original code and not based on any other software
"""

from ..mathutils.czt import czt
import numpy as np


def czt_nf_to_ff(Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray, 
        sampling_distance: float, lambda_vac: float, 
        n_medium: float, n_bfp: float, focal_length: float, NA: float,
        bfp_sampling_n=101
):  
    assert NA <= n_medium, "NA cannot be larger than n_medium"
    Ex = np.atleast_2d(Ex)
    Ey = np.atleast_2d(Ey)
    Ez = np.atleast_2d(Ez)
    assert Ex.shape == Ey.shape == Ez.shape, "All fields need to be equal size"
    assert Ex.shape[0] == Ex.shape[1], "Field matrices need to be square"
    dx = sampling_distance
    
    f = focal_length
    k0 = 2 * np.pi / lambda_vac
    k = k0 * n_medium
    ks = 2 * np.pi / sampling_distance
    a = np.exp(-2j * np.pi * k/ks * NA / n_medium)
    w = np.exp(-4j * np.pi * k/ks * (NA / n_medium) / (bfp_sampling_n-1))

    # The chirp z transform assumes data starting at x[0], but we don't know 
    # where point (0,0) is. Therefore, fix the phases after the
    # transform such that the real and imaginary parts of the fields are what
    # they need to be
    samples_correction = (Ex.shape[0] - 1) / 2
    phase_fix = (a * w**-(np.arange(bfp_sampling_n)))**samples_correction
    phase_fix = phase_fix.reshape((bfp_sampling_n, 1))

    phase_fix_1 = np.tile(phase_fix, (1, Ex.shape[0]))
    phase_fix_2 = np.tile(phase_fix, (1, bfp_sampling_n))

    fEx = np.transpose((czt(Ex, bfp_sampling_n, w, a)) * phase_fix_1, (1,0))
    fEx = czt(fEx, bfp_sampling_n, w, a) * phase_fix_2
    fEy = np.transpose((czt(Ey, bfp_sampling_n, w, a)) * phase_fix_1, (1,0))
    fEy = czt(fEy, bfp_sampling_n, w, a) * phase_fix_2
    fEz = np.transpose((czt(Ez, bfp_sampling_n, w, a)) * phase_fix_1, (1,0))
    fEz = czt(fEz, bfp_sampling_n, w, a) * phase_fix_2
    
    sp = np.linspace(-NA/n_medium, NA/n_medium, bfp_sampling_n)
    Sx, Sy = np.meshgrid(sp, sp)
    Sp = np.hypot(Sx, Sy)
    Sz = np.zeros(Sp.shape)
    Sz[Sp <= NA/n_medium] = (1 - Sp[Sp <= NA/n_medium]**2)**0.5
    
    factor = -(2j * np.pi * k * Sz * np.exp(1j * k * f) * dx**2 
        / (f * 4 * np.pi**2))
    Ex_inf = factor * fEx
    Ey_inf = factor * fEy
    Ez_inf = factor * fEz

    return ff_to_bfp(Ex_inf, Ey_inf, Ez_inf, Sx, Sy, Sz, n_medium, n_bfp)

def ff_to_bfp(Exff: np.ndarray, Eyff: np.ndarray, Ezff: np.ndarray, 
        Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray,
        n_medium: float, n_bfp: float):

    Sp = np.hypot(Sx, Sy)
    cosP = np.ones(Sp.shape)
    sinP = np.zeros(Sp.shape)
    region = Sp > 0
    cosP[region] = Sx[region] / Sp[region]
    sinP[region] = Sy[region] / Sp[region]
    sinP[Sp == 0] = 0
    Et = np.zeros(Exff.shape, dtype='complex128')
    Ep = Et.copy()
    Ex_bfp = Et.copy()
    Ey_bfp = Et.copy()

    Et[Sz > 0]  = ((Exff[Sz > 0]  * cosP[Sz > 0] + 
        Eyff[Sz > 0] * sinP[Sz > 0]) * Sz[Sz > 0] - 
        Ezff[Sz > 0] * Sp[Sz > 0]) * (n_medium/ n_bfp)**0.5 / (Sz[Sz > 0])**0.5
    Ep[Sz > 0] = (Eyff[Sz > 0] * cosP[Sz > 0] - 
        Exff[Sz > 0] * sinP[Sz > 0] ) * (n_medium/n_bfp)**0.5 / (Sz[Sz > 0])**0.5
    
    Ex_bfp[Sz > 0] = Et[Sz > 0] * cosP[Sz > 0] - Ep[Sz > 0] * sinP[Sz > 0]
    Ey_bfp[Sz > 0] = Ep[Sz > 0] * cosP[Sz > 0] + Et[Sz > 0] * sinP[Sz > 0]

    return Ex_bfp, Ey_bfp


def ff_to_bfp_angle(Exff: np.ndarray, Eyff: np.ndarray, Ezff: np.ndarray, 
        cosPhi: np.ndarray, sinPhi: np.ndarray, cosTheta: np.ndarray,
        n_medium: float, n_bfp: float):

    Et = np.zeros(Exff.shape, dtype='complex128')
    Ep = Et.copy()
    Ex_bfp = Et.copy()
    Ey_bfp = Et.copy()

    sinTheta = (1 - cosTheta**2)**0.5

    roc = cosTheta > 0  # roc == Region of convergence, avoid division by zero

    Et[roc]  = ((Exff[roc]  * cosPhi[roc] + 
        Eyff[roc] * sinPhi[roc]) * cosTheta[roc] - 
        Ezff[roc] * sinTheta[roc]) * (n_medium / n_bfp)**0.5 / (cosTheta[roc])**0.5
    
    Ep[roc] = (Eyff[roc] * cosPhi[roc] - 
        Exff[roc] * sinPhi[roc] ) * (n_medium / n_bfp)**0.5 / (cosTheta[roc])**0.5
    
    Ex_bfp[roc] = Et[roc] * cosPhi[roc] - Ep[roc] * sinPhi[roc]
    Ey_bfp[roc] = Ep[roc] * cosPhi[roc] + Et[roc] * sinPhi[roc]

    return Ex_bfp, Ey_bfp
    

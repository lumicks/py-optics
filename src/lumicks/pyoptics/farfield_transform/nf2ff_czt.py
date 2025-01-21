"""
References:
    1. Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
       Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193

This implementation is original code and not based on any other software
"""

import math
import numpy as np

from ..mathutils.czt import czt


def nearfield_to_farfield_czt(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    sampling_step: float,
    lambda_vac: float,
    n_medium: float,
    half_angle: float,
    farfield_radius: float,
    farfield_samples: int,
):
    Ex, Ey, Ez = [np.atleast_2d(E) for E in (Ex, Ey, Ez)]

    if sampling_step <= 0.0:
        raise ValueError("Sampling step has to be > 0.0")
    if lambda_vac <= 0.0:
        raise ValueError("The value of lambda_vac has to be > 0.0")
    if n_medium < 1.0:
        raise ValueError("The value of n_medium has to be ≥ 1.0")
    if not 0 < half_angle <= 90.0:
        raise ValueError("The half-angle has to be between 0° and 90°")
    if farfield_radius <= 0.0:
        raise ValueError("The value of far_field_radius has to be > 0.0")
    if farfield_samples <= 1:
        raise ValueError("The number of samples in the far field needs to be > 1")
    if not (Ex.shape == Ey.shape == Ez.shape):
        raise RuntimeError("All fields need to be equal size")
    if Ex.shape[0] != Ex.shape[1]:
        raise RuntimeError("Field matrices need to be square")

    f = farfield_radius
    k0 = 2 * np.pi / lambda_vac
    k = k0 * n_medium
    ks = 2 * np.pi / sampling_step
    sin_theta = math.sin(math.radians(half_angle))
    a = np.exp(-2j * np.pi * k / ks * sin_theta)
    w = np.exp(-4j * np.pi * k / ks * sin_theta / (farfield_samples - 1))

    # The chirp z transform assumes data starting at x[0], but we don't know
    # where point (0,0) is. Therefore, fix the phases after the
    # transform such that the real and imaginary parts of the fields are what
    # they need to be
    samples_correction = (Ex.shape[0] - 1) / 2
    phase_fix = (a * w ** -(np.arange(farfield_samples))) ** samples_correction
    phase_fix = phase_fix.reshape((farfield_samples, 1))

    phase_fix_1 = np.tile(phase_fix, (1, Ex.shape[0]))
    phase_fix_2 = np.tile(phase_fix, (1, farfield_samples))

    fEx, fEy, fEz = [
        czt(
            np.transpose((czt(E, farfield_samples, w, a)) * phase_fix_1, (1, 0)),
            farfield_samples,
            w,
            a,
        )
        * phase_fix_2
        for E in (Ex, Ey, Ez)
    ]

    sp = np.linspace(-sin_theta, sin_theta, farfield_samples)
    Sx, Sy = np.meshgrid(sp, sp)
    Sp = np.hypot(Sx, Sy)
    Sp[Sp > sin_theta] = 0.0
    Sz = np.zeros_like(Sp)
    roc = Sp <= sin_theta
    Sz[roc] = (1 - Sp[roc] ** 2) ** 0.5

    factor = -(2j * np.pi * k * Sz * np.exp(1j * k * f) * sampling_step**2 / (f * 4 * np.pi**2))
    Ex_inf, Ey_inf, Ez_inf = [factor * fE for fE in (fEx, fEy, fEz)]

    return Sx, Sy, Sz, Ex_inf, Ey_inf, Ez_inf


def _czt_nf_to_ff(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    sampling_distance: float,
    lambda_vac: float,
    n_medium: float,
    n_bfp: float,
    focal_length: float,
    NA: float,
    bfp_sampling_n=101,
):
    if NA <= n_medium:
        raise ValueError("The NA has to be ≤ n_medium")
    if n_medium < 1.0:
        raise ValueError("The value of n_medium has to be ≥ 1.0")
    if n_bfp < 1.0:
        raise ValueError("The value of n_bfp has to be ≥ 1.0")
    if focal_length <= 0.0:
        raise ValueError("The value of focal_length has to be > 0.0")
    if lambda_vac <= 0.0:
        raise ValueError("The value of lambda_vac has to be > 0.0")
    Ex = np.atleast_2d(Ex)
    Ey = np.atleast_2d(Ey)
    Ez = np.atleast_2d(Ez)
    if not (Ex.shape == Ey.shape == Ez.shape):
        raise RuntimeError("All fields need to be equal size")
    if Ex.shape[0] != Ex.shape[1]:
        raise RuntimeError("Field matrices need to be square")
    dx = sampling_distance

    f = focal_length
    k0 = 2 * np.pi / lambda_vac
    k = k0 * n_medium
    ks = 2 * np.pi / sampling_distance
    a = np.exp(-2j * np.pi * k / ks * NA / n_medium)
    w = np.exp(-4j * np.pi * k / ks * (NA / n_medium) / (bfp_sampling_n - 1))

    # The chirp z transform assumes data starting at x[0], but we don't know
    # where point (0,0) is. Therefore, fix the phases after the
    # transform such that the real and imaginary parts of the fields are what
    # they need to be
    samples_correction = (Ex.shape[0] - 1) / 2
    phase_fix = (a * w ** -(np.arange(bfp_sampling_n))) ** samples_correction
    phase_fix = phase_fix.reshape((bfp_sampling_n, 1))

    phase_fix_1 = np.tile(phase_fix, (1, Ex.shape[0]))
    phase_fix_2 = np.tile(phase_fix, (1, bfp_sampling_n))

    fEx = np.transpose((czt(Ex, bfp_sampling_n, w, a)) * phase_fix_1, (1, 0))
    fEx = czt(fEx, bfp_sampling_n, w, a) * phase_fix_2
    fEy = np.transpose((czt(Ey, bfp_sampling_n, w, a)) * phase_fix_1, (1, 0))
    fEy = czt(fEy, bfp_sampling_n, w, a) * phase_fix_2
    fEz = np.transpose((czt(Ez, bfp_sampling_n, w, a)) * phase_fix_1, (1, 0))
    fEz = czt(fEz, bfp_sampling_n, w, a) * phase_fix_2

    sp = np.linspace(-NA / n_medium, NA / n_medium, bfp_sampling_n)
    Sx, Sy = np.meshgrid(sp, sp)
    Sp = np.hypot(Sx, Sy)
    Sp[Sp > NA / n_medium] = 0.0
    Sz = np.zeros_like(Sp)
    roc = Sp <= NA / n_medium
    Sz[roc] = (1 - Sp[roc] ** 2) ** 0.5

    factor = -(2j * np.pi * k * Sz * np.exp(1j * k * f) * dx**2 / (f * 4 * np.pi**2))
    Ex_inf, Ey_inf, Ez_inf = [factor * fE for fE in (fEx, fEy, fEz)]

    return ff_to_bfp(Ex_inf, Ey_inf, Ez_inf, Sx, Sy, Sz, n_medium, n_bfp)


def _ff_to_bfp(
    Exff: np.ndarray,
    Eyff: np.ndarray,
    Ezff: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    Sz: np.ndarray,
    n_medium: float,
    n_bfp: float,
):
    Sp = np.hypot(Sx, Sy)
    cosP = np.ones(Sp.shape)
    sinP = np.zeros(Sp.shape)
    region = Sp > 0
    cosP[region] = Sx[region] / Sp[region]
    sinP[region] = Sy[region] / Sp[region]
    sinP[Sp == 0] = 0
    Et = np.zeros(Exff.shape, dtype="complex128")
    Ep = np.zeros_like(Et)
    Ex_bfp = np.zeros_like(Et)
    Ey_bfp = np.zeros_like(Et)

    roc = Sz > 0
    Et[roc] = (
        ((Exff[roc] * cosP[roc] + Eyff[roc] * sinP[roc]) * Sz[roc] - Ezff[roc] * Sp[roc])
        * (n_medium / n_bfp) ** 0.5
        / (Sz[roc]) ** 0.5
    )
    Ep[roc] = (
        (Eyff[roc] * cosP[roc] - Exff[roc] * sinP[roc])
        * (n_medium / n_bfp) ** 0.5
        / (Sz[roc]) ** 0.5
    )

    Ex_bfp[roc] = Et[roc] * cosP[roc] - Ep[roc] * sinP[roc]
    Ey_bfp[roc] = Ep[roc] * cosP[roc] + Et[roc] * sinP[roc]

    return Ex_bfp, Ey_bfp


def farfield_to_back_focal_plane(
    Exff: np.ndarray,
    Eyff: np.ndarray,
    Ezff: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    Sz: np.ndarray,
    n_medium: float,
    n_bfp: float,
):
    Sp = np.hypot(Sx, Sy)
    cosP = np.ones(Sp.shape)
    sinP = np.zeros(Sp.shape)
    region = Sp > 0
    cosP[region] = Sx[region] / Sp[region]
    sinP[region] = Sy[region] / Sp[region]
    sinP[Sp == 0] = 0

    return ff_to_bfp_angle(
        Exff=Exff,
        Eyff=Eyff,
        Ezff=Eyff,
        cosPhi=cosP,
        sinPhi=sinP,
        cosTheta=Sz,
        n_medium=n_medium,
        n_bfp=n_bfp,
    )


def ff_to_bfp_angle(
    Exff: np.ndarray,
    Eyff: np.ndarray,
    Ezff: np.ndarray,
    cosPhi: np.ndarray,
    sinPhi: np.ndarray,
    cosTheta: np.ndarray,
    n_medium: float,
    n_bfp: float,
):
    Et = np.zeros(Exff.shape, dtype="complex128")
    Ep = np.zeros_like(Et)
    Ex_bfp = np.zeros_like(Et)
    Ey_bfp = np.zeros_like(Et)

    sinTheta = (1 - cosTheta**2) ** 0.5

    roc = cosTheta > 0  # roc == Region of convergence, avoid division by zero

    Et[roc] = (
        (
            (Exff[roc] * cosPhi[roc] + Eyff[roc] * sinPhi[roc]) * cosTheta[roc]
            - Ezff[roc] * sinTheta[roc]
        )
        * (n_medium / n_bfp) ** 0.5
        / (cosTheta[roc]) ** 0.5
    )

    Ep[roc] = (
        (Eyff[roc] * cosPhi[roc] - Exff[roc] * sinPhi[roc])
        * (n_medium / n_bfp) ** 0.5
        / (cosTheta[roc]) ** 0.5
    )

    Ex_bfp[roc] = Et[roc] * cosPhi[roc] - Ep[roc] * sinPhi[roc]
    Ey_bfp[roc] = Ep[roc] * cosPhi[roc] + Et[roc] * sinPhi[roc]

    return Ex_bfp, Ey_bfp

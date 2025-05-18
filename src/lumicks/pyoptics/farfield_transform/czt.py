"""
References:
    1. Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
       Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193

This implementation is original code and not based on any other software
"""

import math

import numpy as np

from ..mathutils.czt import czt


def farfield_czt(
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
    """Calculate the far field based on the sampled near field via a chirped-z transformation.

    Parameters
    ----------
    Ex : np.ndarray
        The x-component of the sampled near field.
    Ey : np.ndarray
        The y-component of the sampled near field.
    Ez : np.ndarray
        The z-component of the sampled near field.
    sampling_step : float
        Spatial step size with which the near field is sampled. Must be in the same units as the
        wavelength `lambda_vac`.
    lambda_vac : float
        Wavelength of the light in vacuum. Must be the same unit as the sampling step.
    n_medium : float
        Refractive index of the (semi-)infinite medium that the light propagates in.
    half_angle : float
        Half the acceptance angle of the cone of rays, or θ in the formula NA = n sin θ. Radians.
    farfield_radius : float
        Radius at which to calculate the far field, in the same units as the wavelength.
    farfield_samples : int
        Number of samples of the far field to calculate .

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple of Numpy arrays, with the first three elements sx, sy, sz representing the normal
        vectors of the far field, and the second three elements Ex_ff, Ey_ff, Ez_ff representing the
        electric field components of the far field.

    Raises
    ------
    ValueError
        Raised if the sampling step is <= 0.0, if lambda_vac ≤ 0, if n_medium < 1.0, if θ ≤ 0° or θ>
        90°, if field_radius < 0.0 or if the number of samples of the far field is ≤ 1.
    RuntimeError
        Raised if the shapes of the input arrays are not all the same and if the input arrays are
        not square.
    """
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
    sin_theta = math.sin(half_angle)
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

    # Set up the grid of normal vectors that correspond to the field locations
    s_theta = np.linspace(-sin_theta, sin_theta, farfield_samples)
    sx, sy = np.meshgrid(s_theta, s_theta)
    sp = np.hypot(sx, sy)
    sp[sp > sin_theta] = 0.0
    sz = np.zeros_like(sp)
    roc = sp <= sin_theta
    sz[roc] = (1 - sp[roc] ** 2) ** 0.5

    prefactor = -(2j * np.pi * k * sz * np.exp(1j * k * f) * sampling_step**2 / (f * 4 * np.pi**2))

    Ex_ff, Ey_ff, Ez_ff = [
        prefactor
        * czt(
            np.transpose((czt(E, farfield_samples, w, a)) * phase_fix_1, (1, 0)),
            farfield_samples,
            w,
            a,
        )
        * phase_fix_2
        for E in (Ex, Ey, Ez)
    ]

    return sx, sy, sz, Ex_ff, Ey_ff, Ez_ff

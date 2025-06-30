"""
References:
    1. Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
       Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193

This implementation is original code and not based on any other software
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import CZT


def near_field_to_far_field_czt(
    Ex: ArrayLike,
    Ey: ArrayLike,
    Ez: ArrayLike,
    sampling_period: float,
    radius: float,
    acceptance_angle: float,
    lambda_vac: float,
    n_medium: float,
    sampling_distance: float = 0.0,
    farfield_sampling_n=101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the far field based on the sampled near field via a chirped-z transformation.

    Parameters
    ----------
    Ex : ArrayLike
        The x-component of the sampled near field.
    Ey : ArrayLike
        The y-component of the sampled near field.
    Ez : ArrayLike
        The z-component of the sampled near field.
    sampling_period : float
        Spatial step size with which the near field is sampled. Must be in the same units as the
        wavelength `lambda_vac`.
    radius : float
        Radius at which to calculate the far field, in the same units as the wavelength.
    acceptance_angle : float
        Half the acceptance angle of the cone of rays, or θ in the formulaa NA = n sin θ. Half the
        acceptance angle of the cone of rays, or θ in the formula NA = n sin θ. Radians.
    lambda_vac : float
        Wavelength of the light in vacuum. Must be the same unit as the sampling step.
    n_medium : float
        Refractive index of the (semi-)infinite medium that the light propagates in.
    sampling_distance : float
        Distance from the origin where the field is sampled. Used to correct for the fact that the
        transform, when unspecified, assumes that the sampling plane is at the origin.
    farfield_sampling_n : int
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
        Raised if the sampling step is <= 0.0, if lambda_vac ≤ 0, if n_medium < 1.0, if θ ≤ 0.0 or
        θ> π/2, if field_radius < 0.0 or if the number of samples of the far field is ≤ 1.
    RuntimeError
        Raised if the shapes of the input arrays are not all the same and if the input arrays are
        not square.
    """
    Ex, Ey, Ez = [np.atleast_2d(E) for E in (Ex, Ey, Ez)]

    if not 0 <= acceptance_angle <= np.pi / 2:
        raise ValueError("The value for acceptance_angle needs to be between 0 and π/2")

    if not (Ex.shape == Ey.shape == Ez.shape):
        raise RuntimeError("All fields need to be equal size")

    if not (Ex.shape[0] == Ex.shape[1]):
        raise RuntimeError("Field matrices need to be square")
    dx = sampling_period
    sin_acceptance_angle = np.sin(acceptance_angle)

    k0 = 2 * np.pi / lambda_vac
    k = k0 * n_medium
    ks = 2 * np.pi / sampling_period
    a = np.exp(-2j * np.pi * k / ks * sin_acceptance_angle)
    w = np.exp(-4j * np.pi * k / ks * sin_acceptance_angle / (farfield_sampling_n - 1))

    # The chirp z transform assumes data starting at x[0], but that's not where the origin is.
    # Therefore, fix the phases after the transform such that the real and imaginary parts of the
    # fields are what they need to be
    samples_correction = (Ex.shape[0] - 1) / 2
    phase_fix = (a * w ** -(np.arange(farfield_sampling_n))) ** samples_correction
    x_czt = CZT(Ex.shape[0], farfield_sampling_n, w, a)
    y_czt = CZT(Ex.shape[1], farfield_sampling_n, w, a)

    fEx = x_czt(Ex, axis=0) * phase_fix[:, np.newaxis]
    fEx = y_czt(fEx, axis=1) * phase_fix[np.newaxis, :]
    fEy = x_czt(Ey, axis=0) * phase_fix[:, np.newaxis]
    fEy = y_czt(fEy, axis=1) * phase_fix[np.newaxis, :]
    fEz = x_czt(Ez, axis=0) * phase_fix[:, np.newaxis]
    fEz = y_czt(fEz, axis=1) * phase_fix[np.newaxis, :]

    sp = np.linspace(-sin_acceptance_angle, sin_acceptance_angle, farfield_sampling_n)
    Sx, Sy = np.meshgrid(sp, sp)
    Sp = np.hypot(Sx, Sy)
    Sz = np.zeros(Sp.shape)
    Sz = (1 + 0j - Sp**2) ** 0.5

    factor = -(
        2j * np.pi * k * Sz * np.exp(1j * k * radius) * dx**2 / (radius * 4 * np.pi**2)
    ) * np.exp(-1j * k * Sz * sampling_distance)
    factor[Sz.imag != 0.0] = 0.0
    Ex_inf, Ey_inf, Ez_inf = [factor * fE for fE in (fEx, fEy, fEz)]

    return Sx, Sy, Sz, Ex_inf, Ey_inf, Ez_inf

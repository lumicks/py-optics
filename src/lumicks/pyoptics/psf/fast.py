from typing import Tuple, Union

import numpy as np
from scipy.signal import CZT

"""
Functions to calculate point spread functions of focused wavefronts by use of chirped z-transforms.
"""


def fast_gauss(
    lambda_vac: float,
    n_bfp: float,
    n_medium: float,
    focal_length: float,
    filling_factor: float,
    NA: float,
    x_range: Union[float, tuple[float, float]],
    numpoints_x: int,
    y_range: Union[float, tuple[float, float]],
    numpoints_y: int,
    z: Union[float, np.ndarray],
    bfp_sampling_n=125,
    return_grid=False,
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of an
    Gaussian beam, using the angular spectrum of plane waves method, see [1]_, chapter 3. This
    function uses the chirp-z transform for speedy evaluation of the fields in the focus [2]_.

    This function correctly incorporates the polarized nature of light in a focus. In other words,
    the polarization state at the focus includes electric fields in the x, y, and z directions. This
    is an example of how to use the function fast_psf(), which takes an arbitrary field distribution
    on the back focal plane as input.

    Parameters
    ----------
    lambda_vac : float
        Wavelength of the light, in meters.
    n_bfp : float
        Refractive index at the back focal plane of the objective [-]
    n_medium : float
        Refractive index of the medium into which the light is focused [-]
    focused focal_length : float
        Focal length of the objective, in meters.
    filling_factor : float
        Filling factor of the Gaussian beam over the aperture, defined as w0/R. Here, w0 is the
        waist of the Gaussian beam and R is the radius of the aperture. Range 0...Inf
    NA : float
        Numerical Aperture: n_medium * sin(theta_max) of the objective.
    x_range : Union[float, tuple(float, float)]
        Size of the PSF along x, in meters. If the range is a single float, it is centered around
        zero. The algorithm will calculate at x locations [-x_range/2..x_range/2]. Otherwise, it will
        calculate at locations from [x_range[0]..x_range[1]]
    numpoints_x : int
        Number of points to calculate along the x dimension. Must be > 0
    y_range : Union[float, tuple(float, float)]
        Same as x, but along y.
    numpoints_y: int
        Same as x, but for y.
    z : Union[np.array, float]
        Numpy array of locations along z, in meters, where to calculate the fields. Can be a single
        number as well.
    bfp_sampling_n :  int, optional
        number of discrete steps with which the back focal plane is sampled, from the center to the
        edge. The total number of plane waves scales with the square of bfp_sampling_n. Default
        value = 125.
    return_grid : bool, optional
        return the sampling grid (default value = False).

    Returns
    -------
    Ex : ndarray
        the electric field along x, as a function of (x, y, z) [V/m]
    Ey : ndarray
        the electric field along y, as a function of (x, y, z) [V/m]
    Ez : ndarray
        the electric field along z, as a function of (x, y, z) [V/m]

    If return_grid is True, then also return the sampling grid X, Y and Z.

    All results are returned with the minimum number of dimensions required to store the results,
    i.e., sampling along the XZ plane will return the fields as Ex(x,z), Ey(x,z), Ez(x,z)


    .. [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.). Cambridge:
        Cambridge University Press. doi:10.1017/CBO9780511794193
    .. [2] Marcel Leutenegger, Ramachandra Rao, Rainer A. Leitgeb, and Theo Lasser, "Fast focus
       field calculations," Opt. Express 14, 11277-11291 (2006)
    """
    w0 = filling_factor * focal_length * NA / n_medium  # See [1]

    def field_func(_, x_bfp, y_bfp, *args):
        Ein = np.exp(-(x_bfp**2 + y_bfp**2) / w0**2)
        return (Ein, None)

    return fast_psf(
        field_func,
        lambda_vac,
        n_bfp,
        n_medium,
        focal_length,
        NA,
        x_range,
        numpoints_x,
        y_range,
        numpoints_y,
        z,
        bfp_sampling_n,
        return_grid,
    )


def fast_psf(
    f_input_field,
    lambda_vac: float,
    n_bfp: float,
    n_medium: float,
    focal_length: float,
    NA: float,
    x_range: Union[float, Tuple[float, float]],
    numpoints_x: int,
    y_range: Union[float, Tuple[float, float]],
    numpoints_y: int,
    z: np.array,
    bfp_sampling_n=125,
    return_grid=False,
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of an
    arbitrary input field, using the angular spectrum of plane waves method, see [1]_, chapter 3.
    This function uses the chirp-z transform for speedy evaluation of the fields in the focus [2]_.

    This function correctly incorporates the polarized nature of light in a focus. In other words,
    the polarization state at the focus includes electric fields in the x, y, and z directions. The
    input can be any arbitrary polarization in the XY plane.

    Parameters
    ----------
    f_input_field : callable
        Function with signature `f(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n)`, where
        `x_bfp` is a grid of x locations in the back focal plane, determined by the focal length and
        NA of the objective. The corresponding grid for y is `y_bfp`, and `r_bfp` is the radial
        distance from the center of the back focal plane. The float `r_max` is the largest distance
        that falls inside the NA, but `r_bfp` will contain larger numbers as the back focal plane is
        sampled with a square grid. The number of samples from the center to the edge of the NA in
        horizontal or vertical direction is `bfp_sampling_n`. This number is forwarded to the
        callable for convenience. The function must return a tuple `(E_bfp_x, E_bfp_y)`, which are
        the electric fields in the x- and y- direction, respectively, at the sample locations in the
        back focal plane. In other words, `E_bfp_x` describes the electric field of the input beam
        which is polarized along the x-axis. Similarly, `E_bfp_y` describes the y-polarized part of
        the input beam. The fields may be complex, so a phase difference between x and y is
        possible. If only one polarization is used, the other return value must be `None`, e.g., y
        polarization would return `(None, E_bfp_y)`. The fields are post-processed such that any
        part that falls outside of the NA is set to zero.
    lambda_vac : float
        Wavelength of the light [m]
    n_bfp : float
        Refractive index at the back focal plane of the objective [-]
    n_medium : float
        Refractive index of the medium into which the light is focused [-]
    focal_length : float
        Focal length of the objective [m]
    NA : float
        Numerical Aperture = :math:`n_{medium} \\sin(\\theta_{max})`, where :math:`\\theta_{max}` is
        the maximum acceptance angle of the objective [-]
    x_range : Union[float, tuple(float, float)]
        Size of the calculation range along x, in meters. If the range is a single float, it is centered around
        zero. The algorithm will calculate at x locations [-x_range/2..x_range/2]. Otherwise, it will
        calculate at locations from [x_range[0]..x_range[1]]
    numpoints_x : int
        Number of points to calculate along the x dimension. Must be >= 1
    y_range : Union[float, tuple(float, float)]
        Same as x, but along y [m]
    numpoints_y : int
        Same as x, but for y
    z : Union[np.array, float]
        Numpy array of locations along z, where to calculate the fields. Can be a single number as
        well [m]
    bfp_sampling_n : int, optional
        Number of discrete steps with which the back focal plane is sampled, from the center to the
        edge. The total number of plane waves scales with the square of bfp_sampling_n (default =
        125)
    return_grid : bool, optional
        Return the sampling grid (default = False)

    Returns
    -------
    Ex : ndarray
        the electric field along x, as a function of (x, y, z) [V/m]
    Ey : ndarray
        the electric field along y, as a function of (x, y, z) [V/m]
    Ez : ndarray
        the electric field along z, as a function of (x, y, z) [V/m]

    If return_grid is True, then also return the sampling grid X, Y and Z.

    All results are returned with the minimum number of dimensions required to store the results,
    i.e., sampling along the XZ plane will return the fields as Ex(x,z), Ey(x,z), Ez(x,z)


    .. [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.). Cambridge:
        Cambridge University Press. doi:10.1017/CBO9780511794193
    .. [2] Marcel Leutenegger, Ramachandra Rao, Rainer A. Leitgeb, and Theo Lasser, "Fast focus
        field calculations," Opt. Express 14, 11277-11291 (2006)
    """

    def _check_axis(numpts, axrange, axis):
        names = {"x": ("numpoints_x", "x_range"), "y": ("numpoints_y", "y_range")}
        if numpts < 1:
            raise ValueError(f"{names[axis][0]} needs to be >= 1")
        if axrange.size > 2:
            raise RuntimeError(f"{names[axis][1]} needs to be a float or a (min, max) tuple")
        if axrange.size == 1 and numpts > 1:
            raise ValueError(
                f"{names[axis][1]} needs to be a tuple (xmin, xmax) for {names[axis][0]} > 1"
            )
        if axrange.size == 2 and numpts == 1:
            raise ValueError(
                f"{names[axis][1]} needs to be a location (float) for {names[axis][0]} == 1"
            )

    x_range = np.asarray(x_range, dtype="float")
    y_range = np.asarray(y_range, dtype="float")

    _check_axis(numpoints_x, x_range, "x")
    _check_axis(numpoints_y, y_range, "y")

    x_center = np.mean(x_range)
    x_range = 0.0 if x_range.size == 1 else np.abs(np.diff(x_range))
    y_center = np.mean(y_range)
    y_range = 0.0 if y_range.size == 1 else np.abs(np.diff(y_range))
    z = np.atleast_1d(z)

    x_range *= 0.5
    y_range *= 0.5

    k = 2 * np.pi * n_medium / lambda_vac
    ks = k * NA / n_medium

    # M = int(np.max((bfp_sampling_n, 2 * NA**2 * np.max(np.abs(z)) /
    #            (np.sqrt(n_medium**2 - NA**2) * lambda_vac))))

    npupilsamples = 2 * bfp_sampling_n - 1

    dk = ks / (bfp_sampling_n - 1)
    sin_th_max = NA / n_medium
    sin_theta_range = np.zeros(bfp_sampling_n * 2 - 1)
    _sin_theta = np.linspace(0, sin_th_max, num=bfp_sampling_n)
    sin_theta_range[0:bfp_sampling_n] = -_sin_theta[::-1]
    sin_theta_range[bfp_sampling_n:] = _sin_theta[1:]

    sin_theta_x, sin_theta_y = np.meshgrid(sin_theta_range, sin_theta_range, indexing="ij")

    sin_theta = np.hypot(sin_theta_x, sin_theta_y)
    aperture = sin_theta <= sin_th_max

    r_max, x_bfp, y_bfp, r_bfp = [
        sine * focal_length for sine in (sin_th_max, sin_theta_x, sin_theta_y, sin_theta)
    ]

    Einx, Einy = f_input_field(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n)
    if Einx is None and Einy is None:
        raise RuntimeError("Either an x-polarized or a y-polarized input field is required")
    cos_theta = np.ones(sin_theta.shape)
    cos_theta[aperture] = ((1 - sin_theta[aperture]) * (1 + sin_theta[aperture])) ** 0.5

    cos_phi = np.ones_like(sin_theta)
    sin_phi = np.zeros_like(sin_theta)
    region = sin_theta > 0

    cos_phi[region] = sin_theta_x[region] / sin_theta[region]
    cos_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 1
    sin_phi[region] = sin_theta_y[region] / sin_theta[region]
    sin_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 0
    sin_phi[np.logical_not(aperture)] = 0
    cos_phi[np.logical_not(aperture)] = 1

    # Precompute some sines and cosines that are repeatedly used
    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = cos_phi**2 - sin_phi**2
    kz = k * cos_theta

    Einfx_x, Einfy_x, Einfz_x, Einfx_y, Einfy_y, Einfz_y = [0] * 6
    if Einx is not None:
        Einx = np.complex128(Einx)
        Einx[np.logical_not(aperture)] = 0
        Einx *= np.sqrt(n_bfp / n_medium) * np.sqrt(cos_theta) / kz
        Einfx_x = Einx * 0.5 * ((1 - cos_2phi) + (1 + cos_2phi) * cos_theta)
        Einfy_x = Einx * 0.5 * sin_2phi * (cos_theta - 1)
        Einfz_x = cos_phi * sin_theta * Einx
    if Einy is not None:
        Einy = np.complex128(Einy)
        Einy[np.logical_not(aperture)] = 0
        Einy *= np.sqrt(n_bfp / n_medium) * np.sqrt(cos_theta) / kz
        Einfx_y = Einy * 0.5 * sin_2phi * (cos_theta - 1)
        Einfy_y = Einy * 0.5 * ((1 + cos_2phi) + cos_theta * (1 - cos_2phi))
        Einfz_y = Einy * sin_phi * sin_theta

    Einfx = Einfx_x + Einfx_y
    Einfy = Einfy_x + Einfy_y
    Einfz = Einfz_x + Einfz_y

    # Make kz 3D - np.atleast_3d() prepends a dimension, and that is not what we need
    kz = np.reshape(kz, (npupilsamples, npupilsamples, 1))

    Z = np.tile(z, ((2 * bfp_sampling_n - 1), (2 * bfp_sampling_n - 1), 1))
    Exp = np.exp(1j * kz * Z)

    Einfx, Einfy, Einfz = [
        np.tile(np.reshape(E, (npupilsamples, npupilsamples, 1)), (1, 1, z.shape[0])) * Exp
        for E in (Einfx, Einfy, Einfz)
    ]

    # Set up the factors for the chirp z transform
    ax = np.exp(-1j * dk * (x_range - x_center))
    wx = np.exp(-2j * dk * x_range / (numpoints_x - 1)) if numpoints_x > 1 else 1.0
    ay = np.exp(-1j * dk * (y_range - y_center))
    wy = np.exp(-2j * dk * y_range / (numpoints_y - 1)) if numpoints_y > 1 else 1.0

    # The chirp z transform assumes data starting at x[0], but our aperture is
    # symmetric around point (0,0). Therefore, fix the phases after the
    # transform such that the real and imaginary parts of the fields are what
    # they need to be
    phase_fix_x = np.reshape(
        (ax * wx ** -(np.arange(numpoints_x))) ** (bfp_sampling_n - 1),
        (numpoints_x, 1, 1),
    )

    phase_fix_y = np.reshape(
        (ay * wy ** -(np.arange(numpoints_y))) ** (bfp_sampling_n - 1),
        (1, numpoints_y, 1),
    )

    # We break the czt into two steps, as there is an overlap in processing that
    # needs to be done for every polarization. Therefore we can save a bit of
    # overhead by storing the results that can be reused.
    x_czt = CZT(Einfx.shape[0], numpoints_x, wx, ax)
    y_czt = CZT(Einfx.shape[1], numpoints_y, wy, ay)

    Ex, Ey, Ez = [
        y_czt(x_czt(Einf, axis=0) * phase_fix_x, axis=1) * phase_fix_y
        for Einf in (Einfx, Einfy, Einfz)
    ]

    Ex, Ey, Ez = [
        E * -1j * focal_length * np.exp(-1j * k * focal_length) * dk**2 / (2 * np.pi)
        for E in (Ex, Ey, Ez)
    ]

    retval = (np.squeeze(Ex), np.squeeze(Ey), np.squeeze(Ez))

    if return_grid:
        xrange_v = np.linspace(-x_range + x_center, x_range + x_center, numpoints_x)
        yrange_v = np.linspace(-y_range + y_center, y_range + y_center, numpoints_y)
        X, Y, Z = np.meshgrid(xrange_v, yrange_v, np.squeeze(z), indexing="ij")
        retval += (np.squeeze(X), np.squeeze(Y), np.squeeze(Z))

    return retval

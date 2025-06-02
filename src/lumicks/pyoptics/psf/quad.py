import numpy as np
from numba import njit
from numpy.typing import ArrayLike

from lumicks.pyoptics.mathutils.integration import annulus_rule
from lumicks.pyoptics.psf import Objective

"""
Functions to calculate a point spread function of a focused wavefront by direct summation of plane
waves.

References
----------
.. [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.). Cambridge: Cambridge
       University Press. doi:10.1017/CBO9780511794193
.. [2] Marcel Leutenegger, Ramachandra Rao, Rainer A. Leitgeb, and Theo Lasser, "Fast focus field
      calculations," Opt. Express 14, 11277-11291 (2006)
"""


def focus_gaussian_quad(
    objective: Objective,
    lambda_vac: float,
    filling_factor: float,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    integration_order=None,
    return_grid=False,
    method="peirce",
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of a Gaussian beam, using the
    angular spectrum of plane waves method and two-dimensional quadrature. See [1], chapter 3.

    This function correctly incorporates the polarized nature of light in a focus. In other words,
    the polarization state at the focus includes electric fields in the x, y, and z directions. The
    input is taken to be polarized along the x direction. This is an example of how to use the
    function fast_psf_calc(), which takes an arbitrary field distribution on the back focal plane as
    input.

    Parameters
    ----------
    lambda_vac : float
      wavelength of the light [m]
    n_bfp : float
      refractive index at the back focal plane of the objective [-]
    n_medium : float
      refractive index of the medium into which the light is focused [-]
    focal_length : float
      focal length of the objective [m]
    filling_factor : float
      filling factor of the Gaussian beam over the aperture, defined as w0 / R. Here, w0 is the
      waist of the Gaussian beam and R is the radius of the aperture. Range 0...Inf [-]
    NA : float
      Numerical Aperture n_medium * sin(theta_max) of the objective [-]
    x : np.array
      array of x locations for evaluation [m]
    y : np.array
      array of y locations for evaluation [m]
    z : np.array
      array of z locations for evaluation. The final locations are determined by the
      output of numpy.meshgrid(x, y, z) [m]
    integration_order : int
      Number of discrete steps with which the back focal plane is sampled, from the center to the
      edge. The total number of plane waves scales with the square of integration_order. Default is
      None, which means that `objective.mimimal_integration_order()` is used as a base. [-]
    return_grid : bool
      return the coordinate sampling grid. Default is False
    method : str
        Method of integration. Options are "equidistant", which yields the same results as
        `psf.focus_czt` within machine precision, and "peirce", which typical converges much faster
        and therefore needs less points.

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
    """
    w0 = filling_factor * objective.r_bfp_max  # See [1]

    def field_func(_, x_bfp, y_bfp, *args):
        Ein = np.exp(-(x_bfp**2 + y_bfp**2) / w0**2)
        return (Ein, None)

    return focus_quad(
        field_func, objective, lambda_vac, x, y, z, integration_order, return_grid, method
    )


def focus_quad(
    f_input_field,
    objective: Objective,
    lambda_vac: float,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    integration_order=None,
    return_grid=False,
    method="peirce",
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of an
    arbitrary input field, using the angular spectrum of plane waves method, see [1], chapter 3.

    This function correctly incorporates the polarized nature of light in a focus. In other words,
    the polarization state at the focus includes electric fields in the x, y, and z directions.

    This function is not recommended in general, as it's slow. However, the points (x, y, z)  at
    which the point spread function is to be evaluated are not necessarily equally spaced, which is
    more flexible. Furthermore, the direct evaluation  of the transform, that is performed here,
    could be faster for a small number of points, over the overhead of using FFTs. It's mostly used
    to benchmark the fast version of this function, which does use FFTs, and for educational
    purposes.

    Parameters
    ----------
    f_input_field:
        function with signature `f(aperture, x_bfp, y_bfp, r_bfp, r_max)`, where `x_bfp` is a grid
        of x locations in the back focal plane, determined by the focal length and NA of the
        objective. `y_bfp` is the corresponding grid of y locations, and `r_bfp` is the radial
        distance from the center of the back focal plane. `r_max` is the largest distance that falls
        inside the NA, but `r_bfp` will contain larger numbers as the back focal plane is sampled
        with a square grid. The function must return a tuple `(E_bfp_x, E_bfp_y)`, which are the
        electric fields in the x- and y- direction, respectively, at the sample locations in the
        back focal plane. In other words, `E_bfp_x` describes the electric field of the input beam
        which is polarized along the x-axis. Similarly, `E_bfp_y` describes the y-polarized part of
        the input beam. The fields may be complex, so a phase difference between x and y is
        possible. If only one polarization is used, the other return value must be `None`, e.g., y
        polarization would return `(None, E_bfp_y)`. The fields are post-processed such that any
        part that falls outside of the NA is set to zero.
    lambda_vac: float
        wavelength of the light [m]
    n_bfp: float
        refractive index at the back focal plane of the objective [-]
    n_medium: float
        refractive index of the medium into which the light is focused [-]
    focal_length: float
        focal length of the objective [m]
    NA: float
        Numerical Aperture n_medium * sin(theta_max) of the objective [-]
    x: np.array
        array of x locations for evaluation [m]
    y: np.array
        array of y locations for evaluation [m]
    z: np.array:
        array of z locations for evaluation. The final locations are determined by the output of
        `numpy.meshgrid(x, y, z)` [m]
    integration_order : int
      Number of discrete steps with which the back focal plane is sampled, from the center to the
      edge. The total number of plane waves scales with the square of integration_order. Default is
      None, which means that `objective.mimimal_integration_order()` is used as a base. [-]
    return_grid: bool
        return the coordinate sampling grid (default = `False`)
    method : str
        Method of integration. Options are "equidistant", which yields the same results as
        `psf.focus_czt` within machine precision, and "peirce", which typical converges much faster
        and therefore needs less points.

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

    """

    # Generate the grid on which to evaluate the PSF, i.e., the sampling of the PSF
    X, Y, Z = np.meshgrid(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z), indexing="ij")

    k = 2 * np.pi * objective.n_medium / lambda_vac
    ks = k * objective.sin_theta_max
    if method not in ["peirce", "equidistant"]:
        raise ValueError("The argument `method` needs to be one of 'peirce' or 'equidistant'")
    if method == "equidistant":
        if integration_order is None:
            integration_order = (
                objective.minimal_integration_order([x, y, z], lambda_vac, "equidistant") * 5
            )  # 5 times oversampling to stay consistent with psf.czt.focus_czt
        npupilsamples = 2 * integration_order - 1
        dk = ks / (integration_order - 1)

        sin_th_max = objective.sin_theta_max
        sin_theta_range = np.zeros(npupilsamples)
        _sin_theta_range = np.linspace(0, sin_th_max, num=integration_order)
        sin_theta_range[0:integration_order] = -_sin_theta_range[::-1]
        sin_theta_range[integration_order:] = _sin_theta_range[1:]
        sin_theta_x, sin_theta_y = np.meshgrid(sin_theta_range, sin_theta_range, indexing="ij")
        sin_theta = np.hypot(sin_theta_x, sin_theta_y)

        # The back focal plane is circular, but our sampling grid is square ->
        # Create a mask: everything outside the NA must be zero
        aperture = sin_theta <= sin_th_max
        r_max = objective.r_bfp_max
        r_bfp = sin_theta * objective.focal_length
        x_bfp = sin_theta_x * objective.focal_length
        y_bfp = sin_theta_y * objective.focal_length

        # Precompute some sines and cosines that are repeatedly used
        cos_theta = np.ones(sin_theta.shape)
        cos_theta[aperture] = (1 - sin_theta[aperture] ** 2) ** 0.5
        cos_phi = np.ones_like(sin_theta)
        sin_phi = np.zeros_like(sin_theta)
        region = sin_theta > 0

        cos_phi[region] = sin_theta_x[region] / sin_theta[region]
        cos_phi[integration_order - 1, integration_order - 1] = 1
        sin_phi[region] = sin_theta_y[region] / sin_theta[region]
        sin_phi[integration_order - 1, integration_order - 1] = 0
        sin_phi[np.logical_not(aperture)] = 0
        cos_phi[np.logical_not(aperture)] = 1
        w = np.ones_like(r_bfp) * dk**2

    if method == "peirce":
        if integration_order is None:
            integration_order = objective.minimal_integration_order(
                [x, y, z], lambda_vac=lambda_vac, method="peirce"
            )
        xi, yi, w = annulus_rule(integration_order, None, 0.0, ks)
        aperture = np.ones_like(xi, dtype=bool)
        r_max = objective.sin_theta_max * objective.focal_length
        sin_theta = np.hypot(xi, yi) / k
        cos_theta = ((1 - sin_theta) * (1 + sin_theta)) ** 0.5
        r_bfp = sin_theta * objective.focal_length
        x_bfp = xi / k * objective.focal_length
        y_bfp = yi / k * objective.focal_length
        cos_phi = x_bfp / r_bfp
        sin_phi = y_bfp / r_bfp

    Einx, Einy = f_input_field(aperture, x_bfp, y_bfp, r_bfp, r_max)
    if Einx is None and Einy is None:
        raise RuntimeError("Either an x-polarized or a y-polarized input field is required")

    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = cos_phi**2 - sin_phi**2
    kz = k * cos_theta

    # TODO: Use Objective.back_focal_plane_to_farfield() + spherical to cartesian
    Einfx_x, Einfy_x, Einfz_x, Einfx_y, Einfy_y, Einfz_y = [0] * 6
    if Einx is not None:
        Einx = np.complex128(Einx)
        Einx[np.logical_not(aperture)] = 0
        Einx *= np.sqrt(objective.n_bfp / objective.n_medium) * np.sqrt(cos_theta) / kz
        Einfx_x = Einx * 0.5 * ((1 - cos_2phi) + (1 + cos_2phi) * cos_theta)
        Einfy_x = Einx * 0.5 * sin_2phi * (cos_theta - 1)
        Einfz_x = cos_phi * sin_theta * Einx
    if Einy is not None:
        Einy = np.complex128(Einy)
        Einy[np.logical_not(aperture)] = 0
        Einy *= np.sqrt(objective.n_bfp / objective.n_medium) * np.sqrt(cos_theta) / kz
        Einfx_y = Einy * 0.5 * sin_2phi * (cos_theta - 1)
        Einfy_y = Einy * 0.5 * ((1 + cos_2phi) + cos_theta * (1 - cos_2phi))
        Einfz_y = Einy * sin_phi * sin_theta

    Einfx = Einfx_x + Einfx_y
    Einfy = Einfy_x + Einfy_y
    Einfz = Einfz_x + Einfz_y

    # Calculate properties of the plane waves
    # As they come from the negative z-direction, a point at infinity with a
    # negative x coordinate leads to a positive value for kx (as the wave is
    # traveling towards point (0,0,0)). Similarly, a negative y coordinate also
    # leads to a positive value for ky
    kp = k * sin_theta
    kx = -kp * cos_phi
    ky = -kp * sin_phi

    Ex, Ey, Ez = _do_loop(aperture, X, Y, Z, w, kx, ky, kz, Einfx, Einfy, Einfz)

    for E in [Ex, Ey, Ez]:
        E *= -1j * objective.focal_length * np.exp(-1j * k * objective.focal_length) / (2 * np.pi)
    retval = (np.squeeze(Ex), np.squeeze(Ey), np.squeeze(Ez))

    if return_grid:
        retval += (np.squeeze(X), np.squeeze(Y), np.squeeze(Z))

    return retval


def _do_loop(aperture, X, Y, Z, w, kx, ky, kz, Einfx, Einfy, Einfz):
    aperture, w, kx, ky, kz, Einfx, Einfy, Einfz = [
        E.reshape(E.size) for E in (aperture, w, kx, ky, kz, Einfx, Einfy, Einfz)
    ]

    # This is tucked in an inner function because Numba objected to the E.flatten above:
    @njit(parallel=False, cache=True)
    def _loop(aperture, X, Y, Z, w, kx, ky, kz, Einfx, Einfy, Einfz):
        # Now the meat: add plane waves from the angles corresponding to the
        # sampling of the back focal plane. This numerically approximates equation
        # 3.33 of [2]

        # Initialize memory for the fields
        Ex, Ey, Ez = [np.zeros_like(X, dtype="complex128") for _ in range(3)]

        items = np.nonzero(aperture)[0]
        for index in items:
            Exp = np.exp(1j * kx[index] * X + 1j * ky[index] * Y + 1j * kz[index] * Z) * w[index]
            Ex += Einfx[index] * Exp
            Ey += Einfy[index] * Exp
            Ez += Einfz[index] * Exp
        return Ex, Ey, Ez

    return _loop(aperture, X, Y, Z, w, kx, ky, kz, Einfx, Einfy, Einfz)

"""
References:
    1. Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
       Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193
    2. Marcel Leutenegger, Ramachandra Rao, Rainer A. Leitgeb, and Theo Lasser,
       "Fast focus field calculations," Opt. Express 14, 11277-11291 (2006)

This implementation is original code and not based on any other software
"""

from . import czt

import numpy as np
import scipy.special as sp
from scipy.integrate import quad


def focused_gauss(
    lambda_vac: float, n_bfp: float, n_medium: float, focal_length: float, filling_factor: float,
    NA: float, x: np.array, y: np.array, z: np.array, bfp_sampling_n=50, return_grid=False
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of a
    Gaussian beam, using the angular spectrum of plane waves method, see [1],
    chapter 3.

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes electric
    fields in the x, y, and z directions. The input is taken to be polarized
    along the x direction.

    This function is not recommended in general, as it's slow. However, the
    points (x, y, z)  at which the point spread function is to be evaluated are
    not necessarily equally spaced, which is more flexible. Furthermore, the
    direct evaluation  of the transform, that is performed here, could be faster
    for a small number of points, over the overhead of using FFTs. It's mostly
    used to benchmark the fast version of this function, which does use FFTs.

    [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
        Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193

    Args:
      lambda_vac: float: wavelength of the light, in meters.
      n_bfp: float: refractive index at the back focal plane of the objective
      n_medium: float: refractive index of the medium into which the light is
        focused
      focal_length: float: focal length of the objective, in meters
      filling_factor: float: filling factor of the Gaussian beam over the
        aperture, defined as w0/R. Here, w0 is the waist of the Gaussian beam
        and R is the radius of the aperture. Range 0...Inf
      NA: float: Numerical Aperture n_medium * sin(theta_max) of the objective
      x: np.array: array of x locations for evaluation, in meters
      y: np.array: array of y locations for evaluation, in meters
      z: np.array: array of z locations for evaluation, in meters
        The final locations are determined by the output of
        numpy.meshgrid(x, y, z)
      bfp_sampling_n: (Default value = 50) Number of discrete steps with which
        the back focal plane is sampled, from the center to the edge. The total
        number of plane waves scales with the square of bfp_sampling_n
      return_grid: (Default value = False) return the sampling grid

    Returns:
      Ex: the electric field along x, as a function of (x, y, z)
      Ey: the electric field along y, as a function of (x, y, z)
      Ez: the electric field along z, as a function of (x, y, z)
      If return_grid is True, then also return the sampling grid X, Y and Z

    """
    w0 = filling_factor * focal_length * NA / n_medium  # See [1]

    def field_func(X_BFP, Y_BFP, *args):
      Ein = np.exp(-(X_BFP**2 + Y_BFP**2)/w0**2)
      return (Ein, None)
    
    return direct_psf_calc(field_func, lambda_vac, n_bfp, n_medium, 
        focal_length, NA, x, y, z, bfp_sampling_n, return_grid)


def direct_psf_calc(
    f_input_field, lambda_vac: float, n_bfp: float, n_medium: float, focal_length: float,
    NA: float, x: np.array, y: np.array, z: np.array, bfp_sampling_n=50, return_grid=False
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of an
    arbitrary input field, using the angular spectrum of plane waves method, 
    see [1], chapter 3.

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes electric
    fields in the x, y, and z directions. This is an example of how to use the 
    function fast_psf_calc(), which takes an arbitrary field distribution on the
    back focal plane as input.

    This function is not recommended in general, as it's slow. However, the
    points (x, y, z)  at which the point spread function is to be evaluated are
    not necessarily equally spaced, which is more flexible. Furthermore, the
    direct evaluation  of the transform, that is performed here, could be faster
    for a small number of points, over the overhead of using FFTs. It's mostly
    used to benchmark  the fast version of this function, which does use FFTs.

    [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
        Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193

    Args:
        f_input_field: function with signature
        f(X_BFP, Y_BFP, R, Rmax, Theta, Phi), where X_BFP is a grid of x
        locations in the back focal plane, determined by the focal length and NA
        of the objective. Y_BFP is the corresponding grid of y locations, and R
        is the radial distance from the center of the back focal plane. Rmax is
        the largest distance that falls inside the NA, but R will contain larger
        numbers as the back focal plane is sampled with a square grid. Theta is
        defined as the angle with the optical axis (z), and Phi is defined as
        the angle between the x and y axis. The function must return a tuple
        (E_BFP_x, E_BFP_y), which are the electric fields in the x- and y-
        direction, respectively, at the sample locations in the back focal
        plane. The fields may be complex, so a phase difference between x and y
        is possible. If only one polarization is used, the other return value
        must be None, e.g., y polarization would return (None, E_BFP_y). The
        fields are post-processed such that any part that falls outside of the
        NA is set to zero.
      lambda_vac: float: wavelength of the light, in meters.
      n_bfp: float: refractive index at the back focal plane of the objective
      n_medium: float: refractive index of the medium into which the light is
        focused
      focal_length: float: focal length of the objective, in meters
      NA: float: Numerical Aperture n_medium * sin(theta_max) of the objective
      x: np.array: array of x locations for evaluation, in meters
      y: np.array: array of y locations for evaluation, in meters
      z: np.array: array of z locations for evaluation, in meters
        The final locations are determined by the output of
        numpy.meshgrid(x, y, z)
      bfp_sampling_n: (Default value = 50) Number of discrete steps with which
        the back focal plane is sampled, from the center to the edge. The total
        number of plane waves scales with the square of bfp_sampling_n
      return_grid: (Default value = False) return the sampling grid

    Returns:
      Ex: the electric field along x, as a function of (x, y, z)
      Ey: the electric field along y, as a function of (x, y, z)
      Ez: the electric field along z, as a function of (x, y, z)
      If return_grid is True, then also return the sampling grid X, Y and Z

    """

    # Generate the grid on which to evaluate the PSF, i.e., the sampling of the
    # PSF

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    Xe, Ye, Ze = np.meshgrid(x, y, z, indexing='ij')

    k = 2*np.pi*n_medium / lambda_vac
    ks = k * NA / n_medium

    # Calculate the minimum sampling of the Back Focal plane according to [2]
    # We take 50x50 plane waves in one quadrant as a minimum, but > 50 is
    # recommended [2]
    #M = int(np.max((bfp_sampling_n, 2 * NA**2 * np.max(np.abs(z)) /
    #            (np.sqrt(n_medium**2 - NA**2) * lambda_vac))))

    npupilsamples = 2 * bfp_sampling_n - 1

    dk = ks / (bfp_sampling_n - 1)
    sin_th_max = NA / n_medium
    x_BFP = np.linspace(-sin_th_max, sin_th_max, num=npupilsamples)
    X_BFP, Y_BFP = np.meshgrid(x_BFP, x_BFP)
    sin_th_BFP = np.hypot(X_BFP, Y_BFP)

    # The back focal plane is circular, but our sampling grid is square ->
    # Create a mask: everything outside the NA must be zero
    aperture = sin_th_BFP > sin_th_max

    # Calculate the angles theta and phi of the far field with the origin
    # Make the argument complex to avoid a warning by np.arcsin()
    Th = np.zeros(sin_th_BFP.shape)
    Th[np.logical_not(aperture)] = np.arcsin(
        sin_th_BFP[np.logical_not(aperture)])
    Phi = np.arctan2(Y_BFP, X_BFP)
    Phi[sin_th_BFP == 0] = 0
    Phi[aperture] = 0

    Rmax = sin_th_max * focal_length
    R = sin_th_BFP * focal_length
    X_BFP *= Rmax
    Y_BFP *= Rmax
    Einx, Einy = f_input_field(X_BFP, Y_BFP, R, Rmax, Th, Phi)
    assert Einx is not None or Einy is not None,\
           "Either an x-polarized or a y-polarized input field is required"
    
    # Precompute some sines and cosines that are repeatedly used
    cosT = np.cos(Th)
    sin2P = np.sin(2*Phi)
    cos2P = np.cos(2*Phi)
    sinT = np.sin(Th)
    Kz = k * cosT

    # There is something funky with the definition of phi and Ez_inf in [1],
    # but the results that we get now are sensible, meaning that curved 
    # wavefronts have the correct sign for x and z
    if Einx is not None:
        Einx[aperture] = 0
        Einx = np.complex128(Einx)
        Einx *= np.sqrt(n_bfp / n_medium) * np.sqrt(cosT)/Kz
        Einfx_x = Einx * 0.5 * ((1 - cos2P) + (1 + cos2P) * cosT)
        Einfy_x = Einx * 0.5 * sin2P * (cosT - 1)
        Einfz_x = np.cos(Phi) * sinT * Einx
    if Einy is not None:
        Einy[aperture] = 0
        Einy = np.complex128(Einy)
        Einy *= np.sqrt(n_bfp / n_medium) * np.sqrt(cosT)/Kz
        Einfx_y = Einy * 0.5 * sin2P * (cosT - 1)
        Einfy_y = Einy * 0.5 * ((1 + cos2P) + cosT * (1 - cos2P))
        Einfz_y = Einy * np.sin(Phi) * sinT

    if Einx is None:
        Einfx = Einfx_y
        Einfy = Einfy_y
        Einfz = Einfz_y
    elif Einy is None:
        Einfx = Einfx_x
        Einfy = Einfy_x
        Einfz = Einfz_x
    else:
        Einfx = Einfx_x + Einfx_y
        Einfy = Einfy_x + Einfy_y
        Einfz = Einfz_x + Einfz_y

    # Calculate properties of the plane waves
    # As they come from the negative z-direction, a point at infinity with a 
    # negative x coordinate leads to a positive value for kx (as the wave is 
    # traveling towards point (0,0,0)). Similarly, a negative y coordinate also 
    # leads to a positive value for ky
    Kz = k * np.cos(Th)
    Kp = k * np.sin(Th)
    Kx = -Kp * np.cos(Phi)
    Ky = -Kp * np.sin(Phi)

    # Initialize memory for the fields
    Ex = np.zeros(Xe.shape, dtype='complex128')
    Ey = np.zeros(Xe.shape, dtype='complex128')
    Ez = np.zeros(Xe.shape, dtype='complex128')

    # Now the meat: add plane waves from the angles corresponding to the
    # sampling of the back focal plane. This numerically approximates equation
    # 3.33 of [2]

    for m in range(npupilsamples):
        for p in range(npupilsamples):
          if aperture[p, m]:  # Skip points outside aperture
            continue

          Exp = np.exp(1j * Kx[p, m] * Xe + 1j * Ky[p, m] * Ye +
                        1j * Kz[p, m] * Ze)
          Ex += Einfx[p, m] * Exp
          Ey += Einfy[p, m] * Exp
          Ez += Einfz[p, m] * Exp

    Ex *= -1j * focal_length*np.exp(-1j * k * focal_length) * dk**2/(2*np.pi)
    Ey *= -1j * focal_length*np.exp(-1j * k * focal_length) * dk**2/(2*np.pi)
    Ez *= -1j * focal_length*np.exp(-1j * k * focal_length) * dk**2/(2*np.pi)

    Ex = np.squeeze(Ex)
    Ey = np.squeeze(Ey)
    Ez = np.squeeze(Ez)

    Xe = np.squeeze(Xe)
    Ye = np.squeeze(Ye)
    Ze = np.squeeze(Ze)

    if return_grid:
      return Ex, Ey, Ez, Xe, Ye, Ze
    else:
      return Ex, Ey, Ez


def focused_gauss_ref(
    lambda_vac: float, n_bfp: float, n_medium: float, focal_length: float, filling_factor: float,
    NA: float, x: np.array, y: np.array, z: np.array
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of a
    Gaussian beam, using the angular spectrum of plane waves method, see [1],
    chapter 3.

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes electric
    fields in the x, y, and z directions.

    This function does not rely on the discretization of the back focal plane of
    the objective. In contrast, it is a semi-analytical expression that
    involves numerical integration over only one coordinate. Therefore, it is
    by far the most accurate numerical evaluation of the point spread function.
    However, it is also slower. This function is useful to assert convergence
    of the results that are obtained with methods that discretize the back
    focal plane.

    [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
        Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193

    Args:
      lambda_vac: float: wavelength of the light, in meters.
      n_bfp: float: refractive index at the back focal plane of the objective
      n_medium: float: refractive index of the medium into which the light is
    focused
      focal_length: float: focal length of the objective
      filling_factor: float: filling factor of the Gaussian beam over the
    aperture, defined as w0/R. Here, w0 is the waist of the Gaussian beam
    and R is the radius of the aperture. Range 0...Inf
      NA: float: Numerical Aperture n_medium * sin(theta_max) of the objective
      x: np.array: array of x locations for evaluation
      y: np.array: array of y locations for evaluation
      z: np.array: array of z locations for evaluation. The final locations are
    determined by the output of numpy.meshgrid(x, y, z)

    Returns:
      Ex: the electric field along x, as a function of (x, y, z)
      Ey: the electric field along y, as a function of (x, y, z)
      Ez: the electric field along z, as a function of (x, y, z)

    """

    x, y, z = np.atleast_1d(x, y, z)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Calculate all (polar) distances r in the grid, as measured from (0,0)
    r_orig = np.hypot(X, Y)
    # Then transform the matrix into a vector
    r = np.reshape(r_orig, (1, -1))

    # Now get the unique numbers in that vector, so we only calculate the
    # integral for unique distances r
    r, idx_r = np.unique(r, return_inverse=True)

    phi = np.arctan2(Y, X)

    k = 2 * np.pi / lambda_vac * n_medium

    th_max = np.arcsin(NA / n_medium)

    # Storage for the results of the integrals
    I0 = np.zeros(r.shape[0], dtype='complex128')
    I1 = np.zeros(r.shape[0], dtype='complex128')
    I2 = np.zeros(r.shape[0], dtype='complex128')

    # Storage for the actual fields
    Ex = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype='complex128')
    Ey = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype='complex128')
    Ez = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype='complex128')

    f0 = filling_factor
    f = focal_length

    for z_idx, zz in enumerate(z):
        for idx, rr in enumerate(r):
            # These are the integrands as defined in [1]
            def __I00(th):
                return (np.exp(-f0**-2 * np.sin(th)**2 / np.sin(th_max)**2) *
                              np.cos(th)**0.5 * np.sin(th) * (1 + np.cos(th)) *
                              sp.jv(0, k * rr * np.sin(th)) * \
                                    np.exp(1j * k * zz * np.cos(th)))

            def __I00r(th):
                return np.real(__I00(th))

            def __I00i(th):
                return np.imag(__I00(th))

            def __I01(th):
                return (np.exp(-f0**-2 * np.sin(th)**2 / np.sin(th_max)**2) *
                              np.cos(th)**0.5 *
                              np.sin(th)**2 * sp.jv(1, k * rr * np.sin(th)) *
                              np.exp(1j * k * zz * np.cos(th)))

            def __I01r(th):
                return np.real(__I01(th))

            def __I01i(th):
                return np.imag(__I01(th))

            def __I02(th):
                return (np.exp(-f0**-2 * np.sin(th)**2 / np.sin(th_max)**2) *
                              np.cos(th)**0.5 * np.sin(th) * (1 - np.cos(th)) *
                              sp.jv(2, k * rr * np.sin(th)) * np.exp(1j * k *
                              zz * np.cos(th)))

            def __I02r(th):
                return np.real(__I02(th))

            def __I02i(th):
                return np.imag(__I02(th))

            I0[idx] = (-1j * k * f / 2 * (n_bfp / n_medium)**0.5 *
                       np.exp(-1j * k * f)) * (quad(__I00r, 0, th_max)[0] +
                                              1j*quad(__I00i, 0, th_max)[0])
            I1[idx] = (-1j * k * f / 2 * (n_bfp / n_medium)**0.5 *
                       np.exp(-1j * k * f)) * (quad(__I01r, 0, th_max)[0] +
                                              1j*quad(__I01i, 0, th_max)[0])
            I2[idx] = (-1j * k * f / 2 * (n_bfp / n_medium)**0.5 *
                       np.exp(-1j * k * f)) * (quad(__I02r, 0, th_max)[0] +
                                              1j*quad(__I02i, 0, th_max)[0])

        # Transform the results back to the grid
        sx = X.shape
        I0_ = np.reshape(I0[idx_r], sx)
        I1_ = np.reshape(I1[idx_r], sx)
        I2_ = np.reshape(I2[idx_r], sx)

        # Calculate the fields
        Ex[:, :, z_idx] = I0_ + I2_ * np.cos(2 * phi)
        Ey[:, :, z_idx] = I2_ * np. sin(2 * phi)
        Ez[:, :, z_idx] = -2j * I1_ * np.cos(phi)

    return np.squeeze(Ex), np.squeeze(Ey), np.squeeze(Ez)


def fast_gauss_psf(
    lambda_vac: float, n_bfp: float, n_medium: float, focal_length: float, filling_factor: float,
    NA: float, xrange: float, numpoints_x: int, yrange:float, numpoints_y: int, z: np.array,
    bfp_sampling_n=125, return_grid=False
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of an
    Gaussian beam, using the angular spectrum of plane waves method, see [1],
    chapter 3. This function uses the chirp-z transform for speedy evaluation of
    the fields in the focus.

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes electric
    fields in the x, y, and z directions. This is an example of how to use the 
    function fast_psf_calc(), which takes an arbitrary field distribution on the
    back focal plane as input.

    Args:
      lambda_vac: float: wavelength of the light, in meters.
      n_bfp: float: refractive index at the back focal plane of the objective
      n_medium: float: refractive index of the medium into which the light is
        focused
      focal_length: float: focal length of the objective, in meters
      filling_factor: float: filling factor of the Gaussian beam over the
        aperture, defined as w0/R. Here, w0 is the waist of the Gaussian beam
        and R is the radius of the aperture. Range 0...Inf
      NA: float: Numerical Aperture n_medium * sin(theta_max) of the objective
      xrange: float: size of the PSF along x, in meters, and centered around
        zero. The algorithm will calculate at x locations [-xrange/2..xrange/2]
      numpoints_x: int: Number of points to calculate along the x dimension.
        Must be > 0
      yrange: float: Same as x, but along y
      numpoints_y: int: Same as x, but for y
      z: np.array: numpy array of locations along z, in meters, where to
        calculate the fields. Can be a single number as well.
      bfp_sampling_n:  (Default value = 125):  Number of discrete steps with
        which the back focal plane is sampled, from the center to the edge.
        The total number of plane waves scales with the square of bfp_sampling_n
      return_grid: (Default value = False): return the sampling grid
    """
    w0 = filling_factor * focal_length * NA / n_medium  # See [1]

    def field_func(X_BFP, Y_BFP, *args):
      Ein = np.exp(-(X_BFP**2 + Y_BFP**2)/w0**2)
      return (Ein, None)

    return fast_psf_calc(field_func, lambda_vac, n_bfp, n_medium, focal_length,
        NA, xrange, numpoints_x, yrange, numpoints_y, z, bfp_sampling_n,
        return_grid)


def fast_psf_calc(
    f_input_field, lambda_vac: float, n_bfp: float, n_medium: float, focal_length: float,
    NA: float, xrange: float, numpoints_x: int, yrange:float, numpoints_y: int, z: np.array,
    bfp_sampling_n=125, return_grid=False
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of an
    arbitrary input field, using the angular spectrum of plane waves method,
    see [1], chapter 3. This function uses the chirp-z transform for speedy
    evaluation of the fields in the focus.

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes electric
    fields in the x, y, and z directions. The input can be any arbitrary
    polarization in the XY plane.

    Args:
      f_input_field: function with signature
        f(X_BFP, Y_BFP, R, Rmax, Theta, Phi), where X_BFP is a grid of x
        locations in the back focal plane, determined by the focal length and NA
        of the objective. Y_BFP is the corresponding grid of y locations, and R
        is the radial distance from the center of the back focal plane. Rmax is
        the largest distance that falls inside the NA, but R will contain larger
        numbers as the back focal plane is sampled with a square grid. Theta is
        defined as the angle with the optical axis (z), and Phi is defined as
        the angle between the x and y axis. The function must return a tuple
        (E_BFP_x, E_BFP_y), which are the electric fields in the x- and y-
        direction, respectively, at the sample locations in the back focal
        plane. The fields may be complex, so a phase difference between x and y
        is possible. If only one polarization is used, the other return value
        must be None, e.g., y polarization would return (None, E_BFP_y). The
        fields are post-processed such that any part that falls outside of the
        NA is set to zero.
      lambda_vac: float: wavelength of the light, in meters.
      n_bfp: float: refractive index at the back focal plane of the objective
      n_medium: float: refractive index of the medium into which the light is
        focused
      focal_length: float: focal length of the objective, in meters
      NA: float: Numerical Aperture = n_medium * sin(theta_max) of the objective
      xrange: float: size of the PSF along x, in meters, and centered around
        zero. The algorithm will calculate at x locations [-xrange/2..xrange/2]
      numpoints_x: int: Number of points to calculate along the x dimension.
        Must be > 0
      yrange: float: Same as x, but along y
      numpoints_y: int: Same as x, but for y
      z: np.array: numpy array of locations along z, in meters, where to
        calculate the fields. Can be a single number as well.
      bfp_sampling_n:  (Default value = 125):  Number of discrete steps with
        which the back focal plane is sampled, from the center to the edge.
        The total number of plane waves scales with the square of bfp_sampling_n
      return_grid: (Default value = False): return the sampling grid

    Returns:
      Ex: the electric field along x, as a function of (x, y, z)
      Ey: the electric field along y, as a function of (x, y, z)
      Ez: the electric field along z, as a function of (x, y, z)
      If return_grid is True, then also return the sampling grid X, Y and Z

      All results are returned with the minimum number of dimensions required to
      store the results, i.e., sampling along the XZ plane will return the
      fields as Ex(x,z), Ey(x,z), Ez(x,z)

    """
    z = np.atleast_1d(z)
    xrange *= 0.5
    yrange *= 0.5

    k = 2*np.pi*n_medium / lambda_vac
    ks = k * NA / n_medium

    #M = int(np.max((bfp_sampling_n, 2 * NA**2 * np.max(np.abs(z)) /
    #            (np.sqrt(n_medium**2 - NA**2) * lambda_vac))))

    npupilsamples = 2*bfp_sampling_n - 1

    dk = ks / (bfp_sampling_n - 1)
    sin_th_max = NA / n_medium
    x_BFP = np.linspace(-sin_th_max, sin_th_max, num=npupilsamples)
    X_BFP, Y_BFP = np.meshgrid(x_BFP, x_BFP, indexing='ij')

    sin_th_BFP = np.hypot(X_BFP, Y_BFP)
    Rmax = focal_length * sin_th_max
    aperture = sin_th_BFP > sin_th_max

    # In contrast to MATLAB, arcsin doesn't like its argument to be > 1, unless
    # you make the type complex explicitly
    Th = np.zeros(sin_th_BFP.shape)
    Th[np.logical_not(aperture)] = np.arcsin(
        sin_th_BFP[np.logical_not(aperture)])
    Phi = np.arctan2(Y_BFP, X_BFP)
    Phi[sin_th_BFP == 0] = 0
    X_BFP *= Rmax
    Y_BFP *= Rmax
    R = sin_th_BFP * focal_length
    Einx, Einy = f_input_field(X_BFP, Y_BFP, R, Rmax, Th, Phi)
    assert Einx is not None or Einy is not None,\
           "Either an x-polarized or a y-polarized input field is required"

    # Precompute some sines and cosines that are repeatedly used
    cosT = np.cos(Th)
    sin2P = np.sin(2*Phi)
    cos2P = np.cos(2*Phi)
    sinT = np.sin(Th)
    Kz = k * cosT

    if Einx is not None:
        Einx[aperture] = 0
        Einx = np.complex128(Einx)
        Einx *= np.sqrt(n_bfp / n_medium) * np.sqrt(cosT)/Kz
        Einfx_x = Einx * 0.5 * ((1 - cos2P) + (1 + cos2P) * cosT)
        Einfy_x = Einx * 0.5 * sin2P * (cosT - 1)
        #  TODO: Auxilliary -1 that needs explanation
        # Current suspicion: definition of theta, phi in [1]
        # Funky things going on with phi for far field
        # Might depend on interpretation of theta - angle with +z or -z axis?
        Einfz_x = np.cos(Phi) * sinT * Einx
    if Einy is not None:
        Einy[aperture] = 0
        Einy = np.complex128(Einy)
        Einy *= np.sqrt(n_bfp / n_medium) * np.sqrt(cosT)/Kz
        Einfx_y = Einy * 0.5 * sin2P * (cosT - 1)
        Einfy_y = Einy * 0.5 * ((1 + cos2P) + cosT * (1 - cos2P))
        Einfz_y = Einy * np.sin(Phi) * sinT

    if Einx is None:
        Einfx = Einfx_y
        Einfy = Einfy_y
        Einfz = Einfz_y
    elif Einy is None:
        Einfx = Einfx_x
        Einfy = Einfy_x
        Einfz = Einfz_x
    else:
        Einfx = Einfx_x + Einfx_y
        Einfy = Einfy_x + Einfy_y
        Einfz = Einfz_x + Einfz_y

    Kz = np.reshape(Kz,(npupilsamples, npupilsamples,1))

    Z = np.tile(z, ((2 * bfp_sampling_n - 1), (2 * bfp_sampling_n - 1), 1))
    Exp = np.exp(1j * Kz * Z)

    Einfx = Einfx.reshape((npupilsamples, npupilsamples, 1))
    Einfx = np.tile(Einfx,(1, 1, z.shape[0])) * Exp

    Einfy = Einfy.reshape((npupilsamples, npupilsamples, 1))
    Einfy = np.tile(Einfy,(1, 1, z.shape[0])) * Exp

    Einfz = Einfz.reshape((npupilsamples, npupilsamples, 1))
    Einfz = np.tile(Einfz,(1, 1, z.shape[0])) * Exp

    # Set up the factors for the chirp z transform
    # TODO: allow arbitrary slices along the XZ and YZ plane, instead of at zero
    if numpoints_x > 1:
        ax = np.exp(-1j*dk*xrange)
        wx = np.exp(-2j*dk*xrange/(numpoints_x - 1))
    else:
        ax = wx = 1.

    if numpoints_y > 1:
        ay = np.exp(-1j*dk*yrange)
        wy = np.exp(-2j*dk*yrange/(numpoints_y - 1))
    else:
        ay = wy = 1.

    # The chirp z transform assumes data starting at x[0], but our aperture is
    # symmetric around point (0,0). Therefore, fix the phases after the
    # transform such that the real and imaginary parts of the fields are what
    # they need to be
    phase_fix_x = np.reshape((ax*wx**-(np.arange(numpoints_x)))**(bfp_sampling_n-1),
                         (numpoints_x, 1, 1))
    phase_fix_step1 = np.tile(phase_fix_x, (1, npupilsamples, Z.shape[2]))

    phase_fix_y = np.reshape((ay*wy**-(np.arange(numpoints_y)))**(bfp_sampling_n-1),
                         (numpoints_y, 1, 1))

    phase_fix_step2 = np.tile(phase_fix_y, (1, numpoints_x, Z.shape[2]))

    # We break the czt into two steps, as there is an overlap in processing that
    # needs to be done for every polarization. Therefore we can save a bit of
    # overhead by storing the results that can be reused.
    precalc_step1 = czt.init_czt(Einfx, numpoints_x, wx, ax)
    Ex = np.transpose(czt.exec_czt(Einfx, precalc_step1) * phase_fix_step1,
                      (1, 0, 2))
    precalc_step2 = czt.init_czt(Ex, numpoints_y, wy, ay)
    Ex = np.transpose(czt.exec_czt(Ex, precalc_step2) * phase_fix_step2,
                      (1, 0, 2))\

    Ey = np.transpose(czt.exec_czt(Einfy, precalc_step1) * phase_fix_step1,
                      (1, 0, 2))
    Ey = np.transpose(czt.exec_czt(Ey, precalc_step2) * phase_fix_step2,
                      (1, 0, 2))

    Ez = np.transpose(czt.exec_czt(Einfz, precalc_step1) * phase_fix_step1,
                      (1, 0, 2))
    Ez = np.transpose(czt.exec_czt(Ez, precalc_step2) * phase_fix_step2,
                      (1, 0, 2))


    Ex *= -1j*focal_length*np.exp(-1j*k*focal_length)*dk**2 / (2*np.pi)
    Ey *= -1j*focal_length*np.exp(-1j*k*focal_length)*dk**2 / (2*np.pi)
    Ez *= -1j*focal_length*np.exp(-1j*k*focal_length)*dk**2 / (2*np.pi)

    Ex = np.squeeze(Ex)
    Ey = np.squeeze(Ey)
    Ez = np.squeeze(Ez)

    if return_grid:
        xrange_v = np.linspace(-xrange, xrange, numpoints_x)
        yrange_v = np.linspace(-yrange, yrange, numpoints_y)
        X, Y, Z = np.meshgrid(xrange_v, yrange_v, np.squeeze(z), indexing='ij')
        X = np.squeeze(X)
        Y = np.squeeze(Y)
        Z = np.squeeze(Z)

        return Ex, Ey, Ez, X, Y, Z
    else:
        return Ex, Ey, Ez

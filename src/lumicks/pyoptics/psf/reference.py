import numpy as np
from scipy.integrate import quad
import scipy.special as sp
from scipy.constants import epsilon_0 as EPS0

"""Reference implementations of specific cases of focused wavefronts. Included are Gaussian beams,
and dipoles in a homogeneous environment in high-NA and paraxial versions."""


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

            def __I01(th):
                return (np.exp(-f0**-2 * np.sin(th)**2 / np.sin(th_max)**2) *
                              np.cos(th)**0.5 *
                              np.sin(th)**2 * sp.jv(1, k * rr * np.sin(th)) *
                              np.exp(1j * k * zz * np.cos(th)))

            def __I02(th):
                return (np.exp(-f0**-2 * np.sin(th)**2 / np.sin(th_max)**2) *
                              np.cos(th)**0.5 * np.sin(th) * (1 - np.cos(th)) *
                              sp.jv(2, k * rr * np.sin(th)) * np.exp(1j * k *
                              zz * np.cos(th)))

            I0[idx] = (-1j * k * f / 2 * (n_bfp / n_medium)**0.5 *
                       np.exp(-1j * k * f)) * quad(__I00, 0, th_max, complex_func=True)[0]
            I1[idx] = (-1j * k * f / 2 * (n_bfp / n_medium)**0.5 *
                       np.exp(-1j * k * f)) * quad(__I01, 0, th_max, complex_func=True)[0]
            I2[idx] = (-1j * k * f / 2 * (n_bfp / n_medium)**0.5 *
                       np.exp(-1j * k * f)) * quad(__I02, 0, th_max, complex_func=True)[0]

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


def focused_dipole_ref(
    dipole_moment, lambda_vac: float, n_bfp: float, n_medium: float, focal_length: float,
    NA: float, focal_length_tube: float, x: np.array, y: np.array, z: np.array, return_grid=False
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
  k_ = 2 * np.pi / lambda_vac * n_bfp

  th_max = np.arcsin(NA / n_medium)

  # Storage for the results of the integrals
  I0 = np.zeros(r.shape[0], dtype='complex128')
  I1 = np.zeros(r.shape[0], dtype='complex128')
  I2 = np.zeros(r.shape[0], dtype='complex128')

  f = focal_length
  f_ = focal_length_tube

  # Storage for the actual fields
  Ex = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype='complex128')
  Ey = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype='complex128')

  for z_idx, zz in enumerate(z):
    for idx, rr in enumerate(r):
        # These are the integrands as defined in [1]
        def __I00r(th):
            return (np.cos(th)**0.5 * np.sin(th) * (1 + np.cos(th)) * 
            sp.jv(0, k_ * rr * np.sin(th) * f/f_) * 
            np.cos(k_ * zz * (1 - 0.5 * (f/f_)**2 * np.sin(th)**2)))

        def __I00i(th):
            return (np.cos(th)**0.5 * np.sin(th) * (1 + np.cos(th)) * 
            sp.jv(0, k_ * rr * np.sin(th) * f/f_) * 
            np.sin(k_ * zz * (1 - 0.5 * (f/f_)**2 * np.sin(th)**2)))

        def __I01r(th):
            return (
              np.cos(th)**0.5 * np.sin(th)**2 * 
              sp.jv(1, k_ * rr * np.sin(th) * f/f_) * 
              np.cos(k_ * zz * (1 - 0.5 * (f/f_)**2 * np.sin(th)**2))
            )

        def __I01i(th):
            return (
              np.cos(th)**0.5 * np.sin(th)**2 * 
              sp.jv(1, k_ * rr * np.sin(th) * f/f_) * 
              np.sin(k_ * zz * (1 - 0.5 * (f/f_)**2 * np.sin(th)**2))
            )

        def __I02r(th):
            return (np.cos(th)**0.5 * np.sin(th) * (1 - np.cos(th)) *
            sp.jv(2, k_ * rr * np.sin(th) * f/f_) * 
            np.cos(k_ * zz * (1 - 0.5 * (f/f_)**2 * np.sin(th)**2))
            )

        def __I02i(th):
            return (np.cos(th)**0.5 * np.sin(th) * (1 - np.cos(th)) *
            sp.jv(2, k_ * rr * np.sin(th) * f/f_) * 
            np.sin(k_ * zz * (1 - 0.5 * (f/f_)**2 * np.sin(th)**2))
            )

        I0[idx] = (quad(__I00r, 0, th_max)[0] +
                    1j * quad(__I00i, 0, th_max)[0])
        I1[idx] = (quad(__I01r, 0, th_max)[0] +
                    1j * quad(__I01i, 0, th_max)[0])
        I2[idx] = (quad(__I02r, 0, th_max)[0] +
                    1j * quad(__I02i, 0, th_max)[0])

    # Transform the results back to the grid
    sx = X.shape
    I0_ = np.reshape(I0[idx_r], sx)
    I1_ = np.reshape(I1[idx_r], sx)
    I2_ = np.reshape(I2[idx_r], sx)

    Ex[:, :, z_idx] += (I0_ + I2_ * np.cos(2 * phi)) * dipole_moment[0]
    Ex[:, :, z_idx] += (I2_ * np.sin(2 * phi)) * dipole_moment[1]
    Ex[:, :, z_idx] += (2j * I1_ * np.cos(phi)) * dipole_moment[2]

    Ey[:, :, z_idx] += (I2_ * np.sin(2 * phi)) * dipole_moment[0]
    Ey[:, :, z_idx] += (I0_ - I2_ * np. cos(2 * phi)) * dipole_moment[1]
    Ey[:, :, z_idx] += (2j * I1_ * np.sin(phi)) * dipole_moment[2]

  factor = (
    (n_medium / n_bfp)**0.5 * 1j * k_ * f * np.exp(1j * (k * f - k_ * f_)) / 
    (8 * np.pi * f_) * (2 * np.pi)**2 / (lambda_vac**2 * EPS0)
  )

  if return_grid:
    return X, Y, np.squeeze(factor * Ex), np.squeeze(factor * Ey)
  else:
    return np.squeeze(factor * Ex), np.squeeze(factor * Ey)


def focused_dipole_paraxial_xy(
    dipole_moment_xy: float, lambda_vac: float, n_bfp: float, n_medium: float, focal_length: float,
    NA: float, focal_length_tube: float, r: np.array
):

    M = focal_length_tube / focal_length * n_medium / n_bfp
    r_ = NA * r / (M * lambda_vac) * 2 * np.pi

    with np.errstate(divide='ignore', invalid='ignore'):
      I = (2 * sp.jv(1, r_) / r_)**2
    I[r_ == 0] = 1.0
    I *= dipole_moment_xy**2 * (NA * np.pi)**4 / (EPS0**2 * n_medium * 
      n_bfp * lambda_vac**6 * M**2)

    return I


def focused_dipole_paraxial_z(
    dipole_moment_z: float, lambda_vac: float, n_bfp: float, n_medium: float, focal_length: float,
    NA: float, focal_length_tube: float, r: np.array
):
    
    M = focal_length_tube / focal_length * n_medium / n_bfp
    r_ = NA * r / (M * lambda_vac) * 2 * np.pi

    with np.errstate(divide='ignore', invalid='ignore'):
      I = (2 * sp.jv(2, r_) / r_)**2
    I[r_ == 0] = 0.0
    I *= dipole_moment_z**2 * (NA / lambda_vac)**6 * np.pi**4 / (EPS0**2 * n_medium**3 * 
      n_bfp * M**2)

    return I

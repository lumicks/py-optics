import numpy as np
import scipy.special as sp
from scipy.constants import epsilon_0 as EPS0
from scipy.integrate import quad

"""Reference implementations of specific cases of focused wavefronts. Included are Gaussian beams,
and dipoles in a homogeneous environment in high-NA and paraxial versions."""


def focused_gauss_ref(
    lambda_vac: float,
    n_bfp: float,
    n_medium: float,
    focal_length: float,
    filling_factor: float,
    NA: float,
    x: np.array,
    y: np.array,
    z: np.array,
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of a
    Gaussian beam, using the angular spectrum of plane waves method, see [1]_, chapter 3.

    This function correctly incorporates the polarized nature of light in a focus. In other words,
    the polarization state at the focus includes electric fields in the x, y, and z directions.

    This function does not rely on the discretization of the back focal plane of the objective. In
    contrast, it is a semi-analytical expression that involves numerical integration over only one
    coordinate. Therefore, it is by far the most accurate numerical evaluation of the point spread
    function. However, it is also slower. This function is useful to assert convergence of the
    results that are obtained with methods that discretize the back focal plane.

    Parameters
    ----------
    lambda_vac : float
        Wavelength of the light, in meters. n_bfp: float: refractive index at the back focal plane
        of the objective n_medium: float: refractive index of the medium into which the light is
        focused
    focal_length : float
        Focal length of the objective in meters
    filling_factor : float
        Filling factor of the Gaussian beam over the aperture, defined as w0/R. Here, w0 is the
        waist of the Gaussian beam and R is the radius of the aperture. Range 0...Inf
    NA : float
        Numerical Aperture n_medium * sin(theta_max) of the objective
    x : np.array
        Array of x locations for evaluation
    y : np.array
        Array of y locations for evaluation
    z : np.array
        Array of z locations for evaluation. The final locations are determined by the output of
        numpy.meshgrid(x, y, z)

    Returns
    -------
    Ex : np.ndarray
        The electric field along x, as a function of (x, y, z)
    Ey : np.ndarray
        The electric field along y, as a function of (x, y, z)
    Ez : np.ndarray
        The electric field along z, as a function of (x, y, z)


    ..  [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
            Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193

    """

    x, y, z = np.atleast_1d(x, y, z)
    X, Y = np.meshgrid(x, y, indexing="ij")

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
    I0 = np.zeros(r.shape[0], dtype="complex128")
    I1 = np.zeros(r.shape[0], dtype="complex128")
    I2 = np.zeros(r.shape[0], dtype="complex128")

    # Storage for the actual fields
    Ex = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype="complex128")
    Ey = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype="complex128")
    Ez = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype="complex128")

    f0 = filling_factor
    f = focal_length

    for z_idx, zz in enumerate(z):
        for idx, rr in enumerate(r):
            # These are the integrands as defined in [1]
            def __I00(th):
                return (
                    np.exp(-(f0**-2) * np.sin(th) ** 2 / np.sin(th_max) ** 2)
                    * (np.cos(th) ** 0.5 * np.sin(th) * (1 + np.cos(th)))
                    * sp.jv(0, k * rr * np.sin(th))
                    * np.exp(1j * k * zz * np.cos(th))
                )

            def __I01(th):
                return (
                    np.exp(-(f0**-2) * np.sin(th) ** 2 / np.sin(th_max) ** 2)
                    * (np.cos(th) ** 0.5 * np.sin(th) ** 2)
                    * sp.jv(1, k * rr * np.sin(th))
                    * np.exp(1j * k * zz * np.cos(th))
                )

            def __I02(th):
                return (
                    np.exp(-(f0**-2) * np.sin(th) ** 2 / np.sin(th_max) ** 2)
                    * (np.cos(th) ** 0.5 * np.sin(th) * (1 - np.cos(th)))
                    * sp.jv(2, k * rr * np.sin(th))
                    * np.exp(1j * k * zz * np.cos(th))
                )

            I0[idx] = (-1j * k * f / 2 * (n_bfp / n_medium) ** 0.5 * np.exp(-1j * k * f)) * quad(
                __I00, 0, th_max, complex_func=True
            )[0]
            I1[idx] = (-1j * k * f / 2 * (n_bfp / n_medium) ** 0.5 * np.exp(-1j * k * f)) * quad(
                __I01, 0, th_max, complex_func=True
            )[0]
            I2[idx] = (-1j * k * f / 2 * (n_bfp / n_medium) ** 0.5 * np.exp(-1j * k * f)) * quad(
                __I02, 0, th_max, complex_func=True
            )[0]

        # Transform the results back to the grid
        sx = X.shape
        I0_ = np.reshape(I0[idx_r], sx)
        I1_ = np.reshape(I1[idx_r], sx)
        I2_ = np.reshape(I2[idx_r], sx)

        # Calculate the fields
        Ex[:, :, z_idx] = I0_ + I2_ * np.cos(2 * phi)
        Ey[:, :, z_idx] = I2_ * np.sin(2 * phi)
        Ez[:, :, z_idx] = -2j * I1_ * np.cos(phi)

    return np.squeeze(Ex), np.squeeze(Ey), np.squeeze(Ez)


def focused_dipole_ref(
    dipole_moment,
    lambda_vac: float,
    n_image: float,
    n_medium: float,
    focal_length: float,
    NA: float,
    focal_length_tube: float,
    x: np.array,
    y: np.array,
    z: np.array,
    return_grid=False,
):
    """Calculate the 3-dimensional, vectorial Point Spread Function of a
    dipole in a homogeneous medium, using the angular spectrum of plane waves method. See [1]_,
    chapter 4.

    This function correctly incorporates the polarized nature of light in a focus. In other words,
    the polarization state at the focus includes electric fields in the x and y directions. The
    field strength in the z direction is considered to be negligible, as it is assumed that the
    focal length of the tube lens is long enough that a paraxial approximation is valid.

    This function does not rely on the discretization of the back focal plane of the objective. In
    contrast, it is a semi-analytical expression that involves numerical integration over only the
    single coordinate theta.

    Parameters
    ----------
    dipole_moment : tuple(px, py, pz)
        Tuple of length 3 that determines the strength and orientation of the diple, in Coulomb
        meter
    lambda_vac : float
        Wavelength of the light, in meters.
    n_image : float
        Refractive index of the medium at the image plane.
    n_medium : float
        Refractive index of the medium where the dipole resides.
    focal_length : float
        Focal length of the objective, in meters
    NA : float
        Numerical Aperture = $n_{medium} \\sin(\\theta_{max})$ of the objective
    focal_length_tube : float
        Focal length of the tube lens, in meters.
    x : np.array
        Array of x locations for evaluation, at the image plane of the tube lens
    y : np.array
        Array of y locations for evaluation, at the image plane of the tube lens
    z : np.array
        Array of z locations for evaluation. The final locations are determined by the output of
        numpy.meshgrid(x, y, z)
    return_grid : bool, optional
        Return the smapling grid that is spanned by x, y and z. By default False.

    Returns
    -------
    Ex : np.ndarray
        The electric field along x, as a function of (x, y, z)
    Ey : np.ndarray
        The electric field along y, as a function of (x, y, z)
    X : np.ndarray
        The coordinates in X (only when `return_grid==True`)
    Y : np.ndarray
        The coordinates in Y (only when `return_grid==True`)
    Z : np.ndarray
        The coordinates in Z (only when `return_grid==True`)


    ..  [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
            Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193

    """

    x, y, z = np.atleast_1d(x, y, z)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Calculate all (polar) distances r in the grid, as measured from (0,0)
    r_orig = np.hypot(X, Y)
    # Then transform the matrix into a vector
    r = np.reshape(r_orig, (1, -1))

    # Now get the unique numbers in that vector, so we only calculate the
    # integral for unique distances r
    r, idx_r = np.unique(r, return_inverse=True)

    phi = np.arctan2(Y, X)

    k = 2 * np.pi / lambda_vac * n_medium
    k_ = 2 * np.pi / lambda_vac * n_image

    th_max = np.arcsin(NA / n_medium)

    # Storage for the results of the integrals
    I0 = np.zeros(r.shape[0], dtype="complex128")
    I1 = np.zeros(r.shape[0], dtype="complex128")
    I2 = np.zeros(r.shape[0], dtype="complex128")

    f = focal_length
    f_ = focal_length_tube

    # Storage for the actual fields
    Ex = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype="complex128")
    Ey = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype="complex128")

    for z_idx, zz in enumerate(z):
        for idx, rr in enumerate(r):
            # These are the integrands as defined in [1]
            def __I00(th):
                return (
                    (np.cos(th) ** 0.5 * np.sin(th) * (1 + np.cos(th)))
                    * sp.jv(0, k_ * rr * np.sin(th) * f / f_)
                    * np.exp(1j * k_ * zz * (1 - 0.5 * (f / f_) ** 2 * np.sin(th) ** 2))
                )

            def __I01(th):
                return (
                    (np.cos(th) ** 0.5 * np.sin(th) ** 2)
                    * sp.jv(1, k_ * rr * np.sin(th) * f / f_)
                    * np.exp(1j * k_ * zz * (1 - 0.5 * (f / f_) ** 2 * np.sin(th) ** 2))
                )

            def __I02(th):
                return (
                    (np.cos(th) ** 0.5 * np.sin(th) * (1 - np.cos(th)))
                    * sp.jv(2, k_ * rr * np.sin(th) * f / f_)
                    * np.exp(1j * k_ * zz * (1 - 0.5 * (f / f_) ** 2 * np.sin(th) ** 2))
                )

            I0[idx] = quad(__I00, 0, th_max, complex_func=True)[0]
            I1[idx] = quad(__I01, 0, th_max, complex_func=True)[0]
            I2[idx] = quad(__I02, 0, th_max, complex_func=True)[0]

        # Transform the results back to the grid
        sx = X.shape
        I0_ = np.reshape(I0[idx_r], sx)
        I1_ = np.reshape(I1[idx_r], sx)
        I2_ = np.reshape(I2[idx_r], sx)

        Ex[:, :, z_idx] += (I0_ + I2_ * np.cos(2 * phi)) * dipole_moment[0]
        Ex[:, :, z_idx] += (I2_ * np.sin(2 * phi)) * dipole_moment[1]
        Ex[:, :, z_idx] += (2j * I1_ * np.cos(phi)) * dipole_moment[2]

        Ey[:, :, z_idx] += (I2_ * np.sin(2 * phi)) * dipole_moment[0]
        Ey[:, :, z_idx] += (I0_ - I2_ * np.cos(2 * phi)) * dipole_moment[1]
        Ey[:, :, z_idx] += (2j * I1_ * np.sin(phi)) * dipole_moment[2]

    factor = (
        (n_medium / n_image) ** 0.5
        * (1j * k_ * f * np.pi)
        * np.exp(1j * (k * f - k_ * f_))
        / (2 * f_ * lambda_vac**2 * EPS0)
    )

    retval = (np.squeeze(factor * Ex), np.squeeze(factor * Ey))
    if return_grid:
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        retval += (X, Y, Z)
    return retval


def focused_dipole_paraxial_xy(
    dipole_moment_xy: float,
    lambda_vac: float,
    n_image: float,
    n_medium: float,
    focal_length: float,
    NA: float,
    focal_length_tube: float,
    r: np.array,
):
    """Calculate the point spread function of a focused dipole in the xy-plane,
    in the paraxial approximation. In that case, the point spread function for x- and y-oriented
    dipoles is the same.

    Parameters
    ----------
    dipole_moment_xy : float
        dipole moment in [Cm]
    lambda_vac : float
        wavelength [m]
    n_image : float
        Refractive index of the medium at the focal plane of the tube lens [-].
    n_medium : float
        Refractive index of the medium that the dipole resides in [-].
    focal_length : float
        focal length of the objective [m].
    NA : float
        Numerical aperture of the objective
    focal_length_tube : float
        focal length of the tube lens [m].
    r : np.array
        radial distance [m].

    Returns
    -------
    squared_electric_field : np.ndarray
        The squared magnitude of the electric field in [V^2/m^2], for dipoles oriented in the plane.
    """
    M = focal_length_tube / focal_length * n_medium / n_image
    r_ = NA * r / (M * lambda_vac) * 2 * np.pi

    with np.errstate(divide="ignore", invalid="ignore"):
        squared_electric_field = (2 * sp.jv(1, r_) / r_) ** 2
    squared_electric_field[r_ == 0] = 1.0
    squared_electric_field *= (dipole_moment_xy**2 * (NA * np.pi) ** 4) / (
        EPS0**2 * n_medium * n_image * lambda_vac**6 * M**2
    )

    return squared_electric_field


def focused_dipole_paraxial_z(
    dipole_moment_z: float,
    lambda_vac: float,
    n_bfp: float,
    n_medium: float,
    focal_length: float,
    NA: float,
    focal_length_tube: float,
    r: np.array,
):
    """Calculate the point spread function of a focused dipole oriented along the z-axis,
    in the paraxial approximation.

    Parameters
    ----------
    dipole_moment_z : float
        dipole moment in [Cm]
    lambda_vac : float
        wavelength [m]
    n_image : float
        Refractive index of the medium at the focal plane of the tube lens [-].
    n_medium : float
        Refractive index of the medium that the dipole resides in [-].
    focal_length : float
        focal length of the objective [m].
    NA : float
        Numerical aperture of the objective
    focal_length_tube : float
        focal length of the tube lens [m].
    r : np.array
        radial distance [m].

    Returns
    -------
    squared_electric_field : np.ndarray
        The squared magnitude of the electric field [V^2/m^2]
    """
    M = focal_length_tube / focal_length * n_medium / n_bfp
    r_ = NA * r / (M * lambda_vac) * 2 * np.pi

    with np.errstate(divide="ignore", invalid="ignore"):
        squared_electric_field = (2 * sp.jv(2, r_) / r_) ** 2
    squared_electric_field[r_ == 0] = 0.0
    squared_electric_field *= (dipole_moment_z**2 * (NA / lambda_vac) ** 6 * np.pi**4) / (
        EPS0**2 * n_medium**3 * n_bfp * M**2
    )
    return squared_electric_field


def reflected_focused_gaussian(
    lambda_vac: float,
    n_bfp: float,
    n_medium: float,
    objective_focal_length: float,
    tube_lens_focal_length: float,
    filling_factor: float,
    NA: float,
    x: np.array,
    y: np.array,
    z: np.array,
):
    x, y, z = np.atleast_1d(x, y, z)
    X, Y = np.meshgrid(x, y, indexing="ij")

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
    I0 = np.zeros(r.shape[0], dtype="complex128")
    I1 = np.zeros(r.shape[0], dtype="complex128")
    I2 = np.zeros(r.shape[0], dtype="complex128")

    # Storage for the actual fields
    Ex = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype="complex128")
    Ey = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype="complex128")
    Ez = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype="complex128")

    f0 = filling_factor
    f = objective_focal_length

    for z_idx, zz in enumerate(z):
        for idx, rr in enumerate(r):
            # These are the integrands as defined in [1]
            def __I00(th):
                return (
                    np.exp(-(f0**-2) * np.sin(th) ** 2 / np.sin(th_max) ** 2)
                    * (np.cos(th) ** 0.5 * np.sin(th) * (1 + np.cos(th)))
                    * sp.jv(0, k * rr * np.sin(th))
                    * np.exp(1j * k * zz * np.cos(th))
                )

            def __I01(th):
                return (
                    np.exp(-(f0**-2) * np.sin(th) ** 2 / np.sin(th_max) ** 2)
                    * (np.cos(th) ** 0.5 * np.sin(th) ** 2)
                    * sp.jv(1, k * rr * np.sin(th))
                    * np.exp(1j * k * zz * np.cos(th))
                )

            def __I02(th):
                return (
                    np.exp(-(f0**-2) * np.sin(th) ** 2 / np.sin(th_max) ** 2)
                    * (np.cos(th) ** 0.5 * np.sin(th) * (1 - np.cos(th)))
                    * sp.jv(2, k * rr * np.sin(th))
                    * np.exp(1j * k * zz * np.cos(th))
                )

            I0[idx] = (-1j * k * f / 2 * (n_bfp / n_medium) ** 0.5 * np.exp(-1j * k * f)) * quad(
                __I00, 0, th_max, complex_func=True
            )[0]
            I1[idx] = (-1j * k * f / 2 * (n_bfp / n_medium) ** 0.5 * np.exp(-1j * k * f)) * quad(
                __I01, 0, th_max, complex_func=True
            )[0]
            I2[idx] = (-1j * k * f / 2 * (n_bfp / n_medium) ** 0.5 * np.exp(-1j * k * f)) * quad(
                __I02, 0, th_max, complex_func=True
            )[0]

        # Transform the results back to the grid
        sx = X.shape
        I0_ = np.reshape(I0[idx_r], sx)
        I1_ = np.reshape(I1[idx_r], sx)
        I2_ = np.reshape(I2[idx_r], sx)

        # Calculate the fields
        Ex[:, :, z_idx] = I0_ + I2_ * np.cos(2 * phi)
        Ey[:, :, z_idx] = I2_ * np.sin(2 * phi)
        Ez[:, :, z_idx] = -2j * I1_ * np.cos(phi)

    return np.squeeze(Ex), np.squeeze(Ey), np.squeeze(Ez)

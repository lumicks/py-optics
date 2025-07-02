import numpy as np
from numpy.typing import ArrayLike


def cosines_from_unit_vectors(sx: ArrayLike, sy: ArrayLike, sz: ArrayLike, normalize: bool = False):
    """Calculate the sines and cosines involved in creating the set of normalized vectors given by
    the coordinates (sx, sy, sz), and equivalent to (cos(φ) * sin(θ), sin(φ) * sin(θ), cos(θ)). Here
    it is assumed that φ is the angle with the x-axis, and θ the angle with the positive z-axis. The
    domain of θ = [0...π] and the domain of φ = [0...2π].

    Parameters
    ----------
    sx : ArrayLike
        Coordinate on the x-axis of a unit vector
    sy : ArrayLike
        Coordinate on the y-axis of a unit vector
    sz : ArrayLike
        Coordinate on the z-axis of a unit vector
    normalize : bool, optional
        Boolean to indicate whether or not the vectors need normalization first, default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns the values for cos_theta, sin_theta, cos_phi and sin_phi, respectively.

    Raises
    ------
    ValueError
        Raised if the input is not a unit vector, or an array of unit vectors (norm != 1.0), if
        `normalize == False`.

    Notes
    -----
    For θ == 0.0, the angle φ is undefined. By convention, this function return cos_phi = 1.0 and
    sin_phi = 0.0.
    """
    sx, sy, sz = [np.atleast_1d(ax) for ax in (sx, sy, sz)]
    r = np.hypot(np.hypot(sx, sy), sz)
    if not normalize and not np.allclose(r, 1.0):
        raise ValueError("The input does not seem to be a unit vector or an array of unit vectors")
    sx, sy, sz = [np.divide(ax, r, out=ax) for ax in (sx, sy, sz)]
    cos_theta = sz
    sin_theta = ((1 + cos_theta) * (1 - cos_theta)) ** 0.5
    sp = np.hypot(sx, sy)
    cos_phi = np.ones_like(sp)
    sin_phi = np.zeros_like(sp)
    region = sin_theta > 0
    cos_phi[region] = sx[region] / sin_theta[region]
    sin_phi[region] = sy[region] / sin_theta[region]
    return cos_theta, sin_theta, cos_phi, sin_phi


def unit_vectors_from_cosines(
    cos_theta: ArrayLike, sin_theta: ArrayLike, cos_phi: ArrayLike, sin_phi: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the unit vectors from the sine and cosines of the corresponding angles. The domain
    of θ is [0..π], and of φ is [0..2π].

    Parameters
    ----------
    cos_theta : ArrayLike
        Cosine of the angle with the z-axis.
    sin_theta : ArrayLike
        Sine of the angle with the z-axis.
    cos_phi : ArrayLike
        Cosine of the angle with the x-axis.
    sin_phi : ArrayLike
        Sine of the angle with the x-axis.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Unit vector as a tuple (sx, sy, sz)

    Raises
    ------
    ValueError
        Raised if the input does not lie on the unit circle(s): cos²θ + sin²θ != 1, and cos²φ +
        sin²φ != 1, and raised if sin θ < 0.0.
    """
    cos_theta, sin_theta, cos_phi, sin_phi = [
        np.asarray(trig) for trig in (cos_theta, sin_theta, cos_phi, sin_phi)
    ]
    if not (
        np.allclose(cos_theta**2 + sin_theta**2, 1.0) and np.allclose(cos_phi**2 + sin_phi**2, 1.0)
    ):
        raise ValueError("The input does not lie on the unit circle")
    if np.any(sin_theta < 0.0):
        raise ValueError("The value of sin_theta cannot be less than zero.")
    sp = sin_theta
    sx = sp * cos_phi
    sy = sp * sin_phi
    return sx, sy, cos_theta


def spherical_to_cartesian(
    locations: tuple[ArrayLike, ArrayLike, ArrayLike],
    f_radial: float | ArrayLike,
    f_theta: float | ArrayLike,
    f_phi: float | ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert farfield spherical field components to cartesian format

    Parameters
    ----------
    locations : tuple[np.ndarray, np.ndarray, np.ndarray]
        List of Numpy arrays describing the (x, y, z) Locations at which E_theta and E_phi are
        taken.
    f_radial : float | ArrayLike
        FIeld component in the radial direction
    f_theta : float | ArrayLike
        Field component in the theta direction
    f_phi : float | ArrayLike
        Field component in the phi direction

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The field components in the x-, y- and z-direction.
    """
    x, y, z = locations
    cos_theta, sin_theta, cos_phi, sin_phi = cosines_from_unit_vectors(x, y, z, normalize=True)
    f_x, f_y, f_z = spherical_to_cartesian_from_angles(
        cos_theta, sin_theta, cos_phi, sin_phi, f_radial, f_theta, f_phi
    )

    return f_x, f_y, f_z


def spherical_to_cartesian_from_angles(
    cos_theta: ArrayLike,
    sin_theta: ArrayLike,
    cos_phi: ArrayLike,
    sin_phi: ArrayLike,
    f_radial: ArrayLike,
    f_theta: ArrayLike,
    f_phi: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_radial, f_theta, f_phi = [np.asarray(field) for field in (f_radial, f_theta, f_phi)]
    cos_theta, sin_theta, cos_phi, sin_phi = [
        np.asarray(trig) for trig in (cos_theta, sin_theta, cos_phi, sin_phi)
    ]
    f_x = f_radial * sin_theta * cos_phi + f_theta * cos_theta * cos_phi - f_phi * sin_phi
    f_y = f_radial * sin_theta * sin_phi + f_theta * cos_theta * sin_phi + f_phi * cos_phi
    f_z = f_radial * cos_theta - f_theta * sin_theta
    return f_x, f_y, f_z


def outer_product(x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: replace with Numpy equivalent
    x, y, z = x1
    vx, vy, vz = x2

    px = y * vz - z * vy
    py = z * vx - x * vz
    pz = x * vy - y * vx
    return px, py, pz

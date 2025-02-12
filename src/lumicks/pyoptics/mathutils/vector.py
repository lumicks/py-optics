import numpy as np


def cosines_from_unit_vectors(sx, sy, sz):
    """Calculate the sines and cosines involved in creating the set of normalized vectors given by
    the coordinates (sx, sy, sz), and equivalent to (cos(φ) * sin(θ), sin(φ) * sin(θ), cos(θ)). Here
    it is assumed that φ is the angle with the x-axis, and θ the angle with the positive z-axis. The
    domain of θ = [0...π] and the domain of φ = [0...2π].

    Parameters
    ----------
    sx : float | np.ndarray
        Coordinate on the x-axis of a unit vector
    sy : float | np.ndarray
        Coordinate on the y-axis of a unit vector
    sz : float | np.ndarray
        Coordinate on the z-axis of a unit vector

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns the values for cos_theta, sin_theta, cos_phi and sin_phi, respectively.

    Notes
    -----
    For θ == 0.0, the angle φ is undefined. By definition, this function return cos_phi = 1.0 and
    sin_phi = 0.0.
    """
    cos_theta = sz
    sin_theta = ((1 + cos_theta) * (1 - cos_theta)) ** 0.5
    sp = np.hypot(sx, sy)
    cos_phi = np.ones_like(sp)
    sin_phi = np.zeros_like(sp)
    region = sin_theta > 0
    cos_phi[region] = sx[region] / sin_theta[region]
    sin_phi[region] = sy[region] / sin_theta[region]
    return cos_theta, sin_theta, cos_phi, sin_phi


def unit_vectors_from_cosines(cos_theta, cos_phi, sin_phi):
    sp = ((1 + cos_theta)(1 - cos_theta)) ** 0.5
    sx = sp * cos_phi
    sy = sp * sin_phi
    return sx, sy


def spherical_to_cartesian(locations, f_radial, f_theta, f_phi):
    """Convert farfield spherical field components to cartesian format

    Parameters
    ----------
    locations : tuple[np.ndarray, np.ndarray, np.ndarray]
        List of Numpy arrays describing the (x, y, z) Locations at which E_theta and E_phi are
        taken.
    f_radial : np.ndarray
        FIeld component in the radial direction
    f_theta : np.ndarray
        Field component in the theta direction
    f_phi : np.ndarray
        Field component in the phi direction

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The field components in the x-, y- and z-direction.
    """
    x, y, z = locations
    r = np.hypot(np.hypot(x, y), z)
    cos_theta, sin_theta, cos_phi, sin_phi = cosines_from_unit_vectors(x / r, y / r, z / r)
    f_x, f_y, f_z = spherical_to_cartesian_from_angles(
        cos_theta, sin_theta, cos_phi, sin_phi, f_radial, f_theta, f_phi
    )

    return f_x, f_y, f_z


def spherical_to_cartesian_from_angles(
    cos_theta, sin_theta, cos_phi, sin_phi, f_radial, f_theta, f_phi
):
    f_x = f_radial * sin_theta * cos_phi + f_theta * cos_theta * cos_phi - f_phi * sin_phi
    f_y = f_radial * sin_theta * sin_phi + f_theta * cos_theta * sin_phi + f_phi * cos_phi
    f_z = f_radial * cos_theta - f_theta * sin_theta
    return f_x, f_y, f_z


def outer_product(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    # TODO: replace with Numpy equivalent
    x, y, z = x1
    vx, vy, vz = x2

    px = y * vz - z * vy
    py = z * vx - x * vz
    pz = x * vy - y * vx
    return px, py, pz

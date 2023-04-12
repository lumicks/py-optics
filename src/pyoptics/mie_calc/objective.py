import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class Objective:
    NA: float
    focal_length: float
    n_bfp: float
    n_medium: float


@dataclass
class BackFocalPlaneCoordinates:
    """Class to store data necessary for doing calculations with the back focal plane"""    
    aperture: np.ndarray
    X_bfp: np.ndarray
    Y_bfp: np.ndarray
    R_bfp: np.ndarray
    R_max: float
    bfp_sampling_n: int


@dataclass
class BackFocalPlaneFields:
    Ex_bfp: np.ndarray
    Ey_bfp: np.ndarray


@dataclass
class FarfieldData:
    """Class to store data necessary for doing calculations with the far field after
    transformation from back focal plane"""
    cos_phi: np.ndarray
    sin_phi: np.ndarray
    cos_theta: np.ndarray
    sin_theta: np.ndarray
    Kx: np.ndarray
    Ky: np.ndarray
    Kz: np.ndarray
    Kp: np.ndarray
    Einf_theta: np.ndarray
    Einf_phi: np.ndarray
    aperture: np.ndarray


def sample_bfp(
    f_input_field, bfp_sampling_n: int, objective: Objective
):
    """Sample f_input_field with bfp_sampling_n samples and return the
    coordinates and the fields as a tuple"""

    sin_theta_max = objective.NA / objective.n_medium
    sin_theta = np.zeros(bfp_sampling_n * 2 - 1)
    _sin_theta = np.linspace(0, sin_theta_max, num=bfp_sampling_n)
    sin_theta[0:bfp_sampling_n] = -_sin_theta[::-1]
    sin_theta[bfp_sampling_n:] = _sin_theta[1:]

    _x_bfp, _y_bfp = np.meshgrid(sin_theta, sin_theta)
    
    X_bfp = _x_bfp * objective.focal_length
    Y_bfp = _y_bfp * objective.focal_length
    R_bfp = np.hypot(_x_bfp, _y_bfp) * objective.focal_length
    R_max = sin_theta_max * objective.focal_length
    aperture = R_bfp <= R_max
    bfp_coords = BackFocalPlaneCoordinates(
        aperture=aperture, X_bfp=X_bfp, Y_bfp=Y_bfp, 
        R_bfp=R_bfp, R_max=R_max, bfp_sampling_n=bfp_sampling_n)
    Ex_bfp, Ey_bfp = f_input_field(**asdict(bfp_coords))

    return bfp_coords, BackFocalPlaneFields(Ex_bfp=Ex_bfp, Ey_bfp=Ey_bfp)


def bfp_to_farfield(
    bfp_coords: BackFocalPlaneCoordinates,
    bfp_fields: BackFocalPlaneFields,
    objective: Objective,
    lambda_vac: float
):
    bfp_sampling_n = bfp_coords.bfp_sampling_n
    sin_theta_max = objective.NA / objective.n_medium
    sin_theta = np.zeros(bfp_sampling_n * 2 - 1)
    _sin_theta = np.linspace(0, sin_theta_max, num=bfp_sampling_n)
    sin_theta[0:bfp_sampling_n] = -_sin_theta[::-1]
    sin_theta[bfp_sampling_n:] = _sin_theta[1:]
    sin_theta_x, sin_theta_y = np.meshgrid(sin_theta, sin_theta)
    sin_theta = np.hypot(sin_theta_x, sin_theta_y)

    # Calculate properties of the plane waves in the far field
    k = 2 * np.pi * objective.n_medium / lambda_vac
    
        
    cos_theta = np.ones(sin_theta.shape)
    cos_theta[bfp_coords.aperture] = (
        1 - sin_theta[bfp_coords.aperture]**2
    )**0.5
    
    cos_phi = np.ones(sin_theta.shape)
    sin_phi = np.zeros(sin_theta.shape)
    region = sin_theta > 0
    cos_phi[region] = (
        bfp_coords.X_bfp[region] / (sin_theta[region] * bfp_coords.R_max)
    )

    cos_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 1
    sin_phi[region] = (
        bfp_coords.Y_bfp[region] / (sin_theta[region] * bfp_coords.R_max)
    )
    sin_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 0
    sin_phi[np.logical_not(bfp_coords.aperture)] = 0
    cos_phi[np.logical_not(bfp_coords.aperture)] = 1
    
    Kz = k * cos_theta
    Kp = k * sin_theta
    Kx = -Kp * cos_phi
    Ky = -Kp * sin_phi
    
    # Transform the input wavefront to a spherical one, after refracting on
    # the Gaussian reference sphere [2], Ch. 3. The field magnitude changes
    # because of the different media, and because of the angle (preservation
    # of power in a beamlet). Finally, incorporate factor 1/kz of the integrand
    if bfp_fields.Ex_bfp is not None:
        Einf_x = np.complex128(
            np.sqrt(objective.n_bfp / objective.n_medium) * 
            bfp_fields.Ex_bfp * np.sqrt(cos_theta) / Kz
        )
        Einf_x[np.logical_not(bfp_coords.aperture)] = 0
    else:
        Einf_x = 0
    
    if bfp_fields.Ey_bfp is not None:
        Einf_y = np.complex128(
            np.sqrt(objective.n_bfp / objective.n_medium) * 
            bfp_fields.Ey_bfp * np.sqrt(cos_theta) / Kz
        )
        Einf_y[np.logical_not(bfp_coords.aperture)] = 0
    else:
        Einf_y = 0

    # Get p- and s-polarized parts
    Einf_theta = (
        Einf_x * cos_phi + Einf_y * sin_phi
    )
    Einf_phi = (
        Einf_y * cos_phi - Einf_x * sin_phi
    )

    return FarfieldData(
        cos_phi=cos_phi, sin_phi=sin_phi, cos_theta=cos_theta, sin_theta=sin_theta,
        Kx=Kx, Ky=Ky, Kz=Kz, Kp=Kp, Einf_theta=Einf_theta, Einf_phi=Einf_phi,
        aperture=bfp_coords.aperture
    )

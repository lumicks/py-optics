import numpy as np
from dataclasses import dataclass

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
    n_pupil_samples: int


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
    bfp_coords = BackFocalPlaneCoordinates()
    bfp_fields = BackFocalPlaneFields()

    npupilsamples = 2 * bfp_sampling_n - 1
    bfp_coords.n_pupil_samples = npupilsamples

    sin_th_max = objective.NA / objective.n_medium
    bfp_coords.R_max = sin_th_max * objective.focal_length
    bfp_coords.aperture = bfp_coords.R_bfp <= bfp_coords.R_max

    _xy_bfp = np.linspace(-sin_th_max, sin_th_max, num=npupilsamples)
    _x_bfp, _y_bfp = np.meshgrid(_xy_bfp, _xy_bfp)
    
    bfp_coords.X_bfp = _x_bfp * objective.focal_length
    bfp_coords.Y_bfp = _y_bfp * objective.focal_length
    bfp_coords.R_bfp = np.hypot(_x_bfp, _y_bfp) * objective.focal_length
    bfp_coords.aperture = bfp_coords.R_bfp <= bfp_coords.R_max
    bfp_fields.Ex_bfp, bfp_fields.Ey_bfp = f_input_field(bfp_coords)

    return bfp_coords, bfp_fields


def bfp_to_farfield(
    bfp_coords: BackFocalPlaneCoordinates,
    bfp_fields: BackFocalPlaneFields,
    objective: Objective,
    lambda_vac: float
):
    farfield = FarfieldData()
    sin_theta_max = objective.NA / objective.n_medium
    _sin_theta = np.linspace(-sin_theta_max, sin_theta_max, num=bfp_coords.n_pupil_samples)
    sin_theta_x, sin_theta_y = np.meshgrid(_sin_theta, _sin_theta)
    farfield.sin_theta = np.hypot(sin_theta_x, sin_theta_y)

    # Calculate properties of the plane waves in the far field
    k = 2 * np.pi * objective.n_medium / lambda_vac
    bfp_sampling_n = (bfp_coords.n_pupil_samples + 1) / 2
        
    farfield.cos_theta = np.ones(farfield.sin_theta.shape)
    farfield.cos_theta[bfp_coords.aperture] = (
        1 - farfield.sin_theta[bfp_coords.aperture]**2
    )**0.5
    
    farfield.cos_phi = np.ones(farfield.sin_theta.shape)
    farfield.sin_phi = np.zeros(farfield.sin_theta.shape)
    region = farfield.sin_theta > 0
    farfield.cos_phi[region] = (
        bfp_coords.X_bfp[region] / (farfield.sin_theta[region] * bfp_coords.R_max)
    )

    farfield.cos_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 1
    farfield.sin_phi[region] = (
        bfp_coords.Y_bfp[region] / (farfield.sin_theta[region] * bfp_coords.R_max)
    )
    farfield.sin_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 0
    farfield.sin_phi[np.logical_not(bfp_coords.aperture)] = 0
    farfield.cos_phi[np.logical_not(bfp_coords.aperture)] = 1
    
    farfield.Kz = k * farfield.cos_theta
    farfield.Kp = k * farfield.sin_theta
    farfield.Kx = -farfield.Kp * farfield.cos_phi
    farfield.Ky = -farfield.Kp * farfield.sin_phi
    farfield.aperture = bfp_coords.aperture
    
    # Transform the input wavefront to a spherical one, after refracting on
    # the Gaussian reference sphere [2], Ch. 3. The field magnitude changes
    # because of the different media, and because of the angle (preservation
    # of power in a beamlet). Finally, incorporate factor 1/kz of the integrand
    if bfp_fields.Ex_bfp is not None:
        Einf_x = np.complex128(
            np.sqrt(objective.n_bfp / objective.n_medium) * 
            bfp_fields.Ex_bfp * np.sqrt(farfield.cos_theta) / farfield.Kz
        )
        Einf_x[np.logical_not(bfp_coords.aperture)] = 0
    else:
        Einf_x = 0
    
    if bfp_fields.Ey_bfp is not None:
        Einf_y = np.complex128(
            np.sqrt(objective.n_bfp / objective.n_medium) * 
            bfp_fields.Ey_bfp * np.sqrt(farfield.cos_theta) / farfield.Kz
        )
        Einf_y[np.logical_not(bfp_coords.aperture)] = 0
    else:
        Einf_y = 0

    # Get p- and s-polarized parts
    farfield.Einf_theta = (
        Einf_x * farfield.cos_phi + Einf_y * farfield.sin_phi
    )
    farfield.Einf_phi = (
        Einf_y * farfield.cos_phi - Einf_x * farfield.sin_phi
    )

    return farfield

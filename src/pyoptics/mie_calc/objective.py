import numpy as np
from dataclasses import dataclass, asdict, astuple

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


@dataclass
class Objective:
    """
    Class to describe the essential properties of an objective
    """
    NA: float
    focal_length: float
    n_bfp: float
    n_medium: float

    @property
    def sin_theta_max(self):
        return self.NA / self.n_medium

    def sine_theta_range(self, bfp_sampling_n):
        sin_theta = np.zeros(bfp_sampling_n * 2 - 1)
        _sin_theta = np.linspace(0, self.sin_theta_max, num=bfp_sampling_n)
        sin_theta[0:bfp_sampling_n] = -_sin_theta[::-1]
        sin_theta[bfp_sampling_n:] = _sin_theta[1:]
        return sin_theta

    def sample_back_focal_plane(
        self, f_input_field, bfp_sampling_n: int
    ):
        """Sample f_input_field with bfp_sampling_n samples and return the
        coordinates and the fields as a tuple"""

        sin_theta = self.sine_theta_range(bfp_sampling_n)        
        _x_bfp, _y_bfp = np.meshgrid(sin_theta, sin_theta)
        
        X_bfp = _x_bfp * self.focal_length
        Y_bfp = _y_bfp * self.focal_length
        sin_theta = np.hypot(_x_bfp, _y_bfp)
        R_bfp =  sin_theta * self.focal_length
        R_max = self.sin_theta_max * self.focal_length
        aperture = sin_theta <= self.sin_theta_max
        bfp_coords = BackFocalPlaneCoordinates(
            aperture=aperture, X_bfp=X_bfp, Y_bfp=Y_bfp, 
            R_bfp=R_bfp, R_max=R_max, bfp_sampling_n=bfp_sampling_n)
        
        bfp_dict = asdict(bfp_coords)
        bfp_dict.pop('aperture')
        Ex_bfp, Ey_bfp = f_input_field(**bfp_dict)

        return bfp_coords, BackFocalPlaneFields(Ex_bfp=Ex_bfp, Ey_bfp=Ey_bfp)

    def back_focal_plane_to_farfield(
        self,
        bfp_coords: BackFocalPlaneCoordinates,
        bfp_fields: BackFocalPlaneFields,
        lambda_vac: float
    ):
        bfp_sampling_n = bfp_coords.bfp_sampling_n
        sin_theta = self.sine_theta_range(bfp_sampling_n)
        sin_theta_x, sin_theta_y = np.meshgrid(sin_theta, sin_theta)
        sin_theta = np.hypot(sin_theta_x, sin_theta_y)

        # Calculate properties of the plane waves in the far field
        k = 2 * np.pi * self.n_medium / lambda_vac
            
        cos_theta = np.ones(sin_theta.shape)
        cos_theta[bfp_coords.aperture] = (
            1 - sin_theta[bfp_coords.aperture]**2
        )**0.5
        
        cos_phi = np.ones_like(sin_theta)
        sin_phi = np.zeros_like(sin_theta)
        region = sin_theta > 0

        cos_phi[region] = (
            sin_theta_x[region] / sin_theta[region]
        )

        cos_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 1
        sin_phi[region] = (
            sin_theta_y[region] / sin_theta[region]
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
        # of power in a beamlet).
        E_inf = []
        for bfp_field in astuple(bfp_fields):
            if bfp_field is not None:
                E = np.complex128(
                    np.sqrt(self.n_bfp / self.n_medium) * 
                    bfp_field * np.sqrt(cos_theta)
                )
                E[np.logical_not(bfp_coords.aperture)] = 0
            else:
                E = 0
            E_inf.append(E)

        # Get p- and s-polarized parts
        Einf_theta = (
            E_inf[0] * cos_phi + E_inf[1] * sin_phi
        )
        Einf_phi = (
            E_inf[1]* cos_phi - E_inf[0] * sin_phi
        )

        return FarfieldData(
            cos_phi=cos_phi, sin_phi=sin_phi, cos_theta=cos_theta, sin_theta=sin_theta,
            Kx=Kx, Ky=Ky, Kz=Kz, Kp=Kp, Einf_theta=Einf_theta, Einf_phi=Einf_phi,
            aperture=bfp_coords.aperture
        )

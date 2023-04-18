import numpy as np
from dataclasses import dataclass, asdict
from collections import namedtuple

@dataclass
class BackFocalPlaneCoordinates:
    """Class to store data necessary for doing calculations with the back focal plane"""    
    aperture: np.ndarray
    x_bfp: np.ndarray
    y_bfp: np.ndarray
    r_bfp: np.ndarray
    r_max: float
    bfp_sampling_n: int


BackFocalPlaneFields = namedtuple('BackFocalPlaneFields', ['Ex', 'Ey'])


@dataclass
class FarfieldData:
    """Class to store data necessary for doing calculations with the far field after
    transformation from back focal plane"""
    cos_phi: np.ndarray
    sin_phi: np.ndarray
    cos_theta: np.ndarray
    sin_theta: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    kp: np.ndarray
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
        x_bfp, y_bfp = np.meshgrid(sin_theta, sin_theta)
        
        sin_theta = np.hypot(x_bfp, y_bfp)
        x_bfp *= self.focal_length
        y_bfp *= self.focal_length
        
        r_bfp =  sin_theta * self.focal_length
        r_max = self.sin_theta_max * self.focal_length
        aperture = sin_theta <= self.sin_theta_max
        bfp_coords = BackFocalPlaneCoordinates(
            aperture=aperture, x_bfp=x_bfp, y_bfp=y_bfp, 
            r_bfp=r_bfp, r_max=r_max, bfp_sampling_n=bfp_sampling_n)
        
        Ex_bfp, Ey_bfp = f_input_field(**asdict(bfp_coords))

        return bfp_coords, BackFocalPlaneFields(Ex=Ex_bfp, Ey=Ey_bfp)

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
        
        kz = k * cos_theta
        kp = k * sin_theta
        kx = -kp * cos_phi
        ky = -kp * sin_phi
        
        # Transform the input wavefront to a spherical one, after refracting on
        # the Gaussian reference sphere [2], Ch. 3. The field magnitude changes
        # because of the different media, and because of the angle (preservation
        # of power in a beamlet).
        E_inf = []
        for bfp_field in bfp_fields:
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
            kx=kx, ky=ky, kz=kz, kp=kp, Einf_theta=Einf_theta, Einf_phi=Einf_phi,
            aperture=bfp_coords.aperture
        )

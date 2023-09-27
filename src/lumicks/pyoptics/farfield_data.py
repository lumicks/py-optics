import numpy as np
from dataclasses import dataclass


@dataclass
class FarfieldData:
    """Class to store data necessary for doing calculations with the far field
    after transformation from back focal plane
    """

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
    
    def transform_to_xyz(self):
        """Transform a far field $E_\\theta$, $E_\\phi$ to cartesian components in x, y and z.

        Returns
        -------
        Ex : np.ndarray
            Array with the electric field in the x direction
        Ey : np.ndarray
            Array with the electric field in the y direction
        Ez : np.ndarray
            Array with the electric field in the z direction
        """        
        Ex = self.Einf_theta * self.cos_phi * self.cos_theta - self.Einf_phi * self.sin_phi
        Ey = self.Einf_theta * self.sin_phi * self.cos_theta + self.Einf_phi * self.cos_phi
        Ez = self.Einf_theta * self.sin_theta
        
        return Ex, Ey, Ez
        


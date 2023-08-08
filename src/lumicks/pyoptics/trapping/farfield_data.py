import numpy as np
from dataclasses import dataclass, fields


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


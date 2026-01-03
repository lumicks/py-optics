from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class PropagationDirection(Enum):
    TO_ORIGIN = auto()
    FROM_ORIGIN = auto()


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
    Einf_theta: np.ndarray
    Einf_phi: np.ndarray
    weights: np.ndarray

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

    def get_unit_vectors(
        self: "FarfieldData",
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """Returns the unit vector (sx, sy, sz) corresponding to FarfieldData.cos_phi,
        FarfieldData.sin_phi, FarfieldData.cos_theta and FarfieldData.sin_theta.

        Returns
        -------
        tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]
            The return values sx, sy, sz
        """
        sz = self.cos_theta
        sx = self.sin_theta * self.cos_phi
        sy = self.sin_theta * self.sin_phi
        return sx, sy, sz


def get_k_vectors_from_unit_vectors(
    unit_vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    k: float,
    direction: PropagationDirection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Calculate properties of the plane waves. As they come from the negative z-direction, a point
    # at infinity with a negative x coordinate leads to a positive value for kx (as the wave is
    # traveling towards point (0,0,0)). Similarly, a negative y coordinate also leads to a positive
    # value for ky
    sx, sy, sz = unit_vectors
    sign = -1 if direction == PropagationDirection.TO_ORIGIN else 1

    kz = k * sz
    kx, ky = [sign * k * s for s in (sx, sy)]
    return kx, ky, kz


def get_k_vectors_from_cosines(
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    k: float,
    direction: PropagationDirection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Calculate properties of the plane waves. As they come from the negative z-direction, a point
    # at infinity with a negative x coordinate leads to a positive value for kx (as the wave is
    # traveling towards point (0,0,0)). Similarly, a negative y coordinate also leads to a positive
    # value for ky
    sign = -1 if direction == PropagationDirection.TO_ORIGIN else 1
    kx, ky, kz = [sign * k * cos_phi * sin_theta, sign * k * sin_phi * sin_theta, k * cos_theta]
    return kx, ky, kz

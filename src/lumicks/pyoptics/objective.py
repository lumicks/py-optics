from collections import namedtuple
from dataclasses import astuple, dataclass
from typing import Tuple

import numpy as np

from .farfield import FarfieldData, PropagationDirection, get_k_vectors_from_cosines
from .mathutils.vector import cosines_from_unit_vectors


@dataclass
class BackFocalPlaneCoordinates:
    """
    Class to store data necessary for doing calculations with the back focal
    plane
    """

    aperture: np.ndarray
    x_bfp: np.ndarray
    y_bfp: np.ndarray
    r_bfp: np.ndarray
    r_max: float
    bfp_sampling_n: int


BackFocalPlaneFields = namedtuple("BackFocalPlaneFields", ["Ex", "Ey"])


class Objective:
    """
    Class to describe the essential properties of an objective.
    """

    def __init__(self, NA: float, focal_length: float, n_bfp: float, n_medium: float):
        """Initialize the Objective class.

        Parameters
        ----------
        NA : float
            Numerical aperture (NA) of the objective. Has to be strictly positive and less than the
            refractive index of the medium `n_medium`.
        focal_length : float
            Focal length of the objective in meters. Has to be strictly positive.
        n_bfp : float
            Refractive index of the medium present at the back-focal plane (BFP) of the objective.
            Has to be strictly positive.
        n_medium : float
            Refractive index of the immersion medium of the objective. Has to be strictly positive
            and larger than the NA.

        Raises
        ------
        ValueError
            Raised if `NA` > `n_medium`, if `n_medium < 0` or `n_bp < 0`, if `focal_length < 0` or
            if `NA < 0`.
        """
        if NA > n_medium:
            raise ValueError("The NA of the objective cannot be larger than n_medium")

        if n_medium <= 0 or n_bfp <= 0:
            raise ValueError(
                "Only positive refractive indices are supported for n_bfp and n_medium"
            )

        if focal_length <= 0:
            raise ValueError("focal_length needs to be strictly positive")

        if NA <= 0:
            raise ValueError("NA needs to be strictly positive")

        self.NA = NA
        self.focal_length = focal_length
        self.n_bfp = n_bfp
        self.n_medium = n_medium

    def __str__(self) -> str:
        return "\n".join(
            [
                "Objective",
                "---------",
                f"NA: {self.NA}",
                f"focal length: {self.focal_length} [m]",
                f"refractive index at back focal plane: {self.n_bfp}",
                f"refractive index of medium: {self.n_medium}",
            ]
        )

    def __repr__(self) -> str:
        return (
            f"Objective(NA={self.NA}, "
            f"focal_length={self.focal_length}, "
            f"n_bfp={self.n_bfp}, "
            f"n_medium={self.n_medium})"
        )

    @property
    def sin_theta_max(self):
        """
        Sine of the maximum acceptance angle of the objective
        """
        return self.NA / self.n_medium

    def sine_theta_range(self, bfp_sampling_n):
        sin_theta = np.zeros(bfp_sampling_n * 2 - 1)
        _sin_theta = np.linspace(0, self.sin_theta_max, num=bfp_sampling_n)
        sin_theta[0:bfp_sampling_n] = -_sin_theta[::-1]
        sin_theta[bfp_sampling_n:] = _sin_theta[1:]
        return sin_theta

    def sample_back_focal_plane(self, f_input_field, bfp_sampling_n: int):
        """
        Sample `f_input_field` with `bfp_sampling_n` samples and return the
        coordinates in a BackFocalPlaneCoordinates object, and the fields as a
        named tuple BackFocalPlaneFields(Ex, Ey).
        """
        bfp_coords = self.get_back_focal_plane_coordinates(bfp_sampling_n)

        Ex_bfp, Ey_bfp = (
            f_input_field(*astuple(bfp_coords)) if f_input_field is not None else (None, None)
        )

        return bfp_coords, BackFocalPlaneFields(Ex=Ex_bfp, Ey=Ey_bfp)

    def get_back_focal_plane_coordinates(self, bfp_sampling_n: int):
        sin_theta = self.sine_theta_range(bfp_sampling_n)
        x_bfp, y_bfp = np.meshgrid(sin_theta, sin_theta, indexing="ij")

        sin_theta = np.hypot(x_bfp, y_bfp)
        x_bfp *= self.focal_length
        y_bfp *= self.focal_length

        r_bfp = sin_theta * self.focal_length
        r_max = self.sin_theta_max * self.focal_length
        aperture = sin_theta <= self.sin_theta_max
        bfp_coords = BackFocalPlaneCoordinates(
            aperture=aperture,
            x_bfp=x_bfp,
            y_bfp=y_bfp,
            r_bfp=r_bfp,
            r_max=r_max,
            bfp_sampling_n=bfp_sampling_n,
        )

        return bfp_coords

    def back_focal_plane_to_farfield(
        self,
        bfp_coords: BackFocalPlaneCoordinates,
        bfp_fields: BackFocalPlaneFields,
        lambda_vac: float,
    ):
        """
        Refract the input beam at a Gaussian reference sphere, while taking
        care that the power in a beamlet is modified according to angle and
        media before and after the reference surface. Returns an instance of
        the `FarfieldData` class.
        """
        cos_theta, sin_theta, cos_phi, sin_phi, aperture = self.get_farfield_cosines(bfp_coords)

        kx, ky, kz = self.get_k_vectors_for_focus(
            lambda_vac, sin_theta, cos_theta, cos_phi, sin_phi
        )

        # Transform the input wavefront to a spherical one, after refracting on
        # the Gaussian reference sphere [2], Ch. 3. The field magnitude changes
        # because of the different media, and because of the angle
        # (preservation of power in a beamlet).
        E_inf = []
        for bfp_field in bfp_fields:
            if bfp_field is not None:
                E = np.complex128(
                    np.sqrt(self.n_bfp / self.n_medium) * bfp_field * np.sqrt(cos_theta)
                )
                E[np.logical_not(aperture)] = 0
            else:
                E = 0
            E_inf.append(E)

        # Get p- and s-polarized parts
        Einf_theta = E_inf[0] * cos_phi + E_inf[1] * sin_phi
        Einf_phi = E_inf[1] * cos_phi - E_inf[0] * sin_phi

        return FarfieldData(
            cos_phi=cos_phi,
            sin_phi=sin_phi,
            cos_theta=cos_theta,
            sin_theta=sin_theta,
            kx=kx,
            ky=ky,
            kz=kz,
            Einf_theta=Einf_theta,
            Einf_phi=Einf_phi,
            aperture=bfp_coords.aperture,
        )

    def get_k_vectors_for_focus(self, lambda_vac, sin_theta, cos_theta, cos_phi, sin_phi):
        k = 2 * np.pi * self.n_medium / lambda_vac

        return get_k_vectors_from_cosines(
            cos_theta, sin_theta, cos_phi, sin_phi, k, PropagationDirection.TO_ORIGIN
        )

    def get_k_vectors_for_scattering(self, lambda_vac, sin_theta, cos_theta, cos_phi, sin_phi):
        k = 2 * np.pi * self.n_medium / lambda_vac

        return get_k_vectors_from_cosines(
            cos_theta, sin_theta, cos_phi, sin_phi, k, PropagationDirection.FROM_ORIGIN
        )

    def get_farfield_cosines(self, bfp_coords: BackFocalPlaneCoordinates):
        bfp_sampling_n = bfp_coords.bfp_sampling_n
        sin_theta_x = bfp_coords.x_bfp / self.focal_length
        sin_theta_y = bfp_coords.y_bfp / self.focal_length
        sin_theta = bfp_coords.r_bfp / self.focal_length * bfp_coords.aperture
        aperture = bfp_coords.aperture  # sin_theta <= self.sin_theta_max
        cos_theta = np.ones(sin_theta.shape)
        cos_theta[aperture] = ((1 + sin_theta[aperture]) * (1 - sin_theta[aperture])) ** 0.5
        cos_phi = np.ones_like(sin_theta)
        sin_phi = np.zeros_like(sin_theta)
        region = sin_theta > 0 & aperture
        cos_phi[region] = sin_theta_x[region] / sin_theta[region]
        cos_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 1
        sin_phi[region] = sin_theta_y[region] / sin_theta[region]
        sin_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 0
        sin_phi[np.logical_not(aperture)] = 0
        cos_phi[np.logical_not(aperture)] = 1
        return cos_theta, sin_theta, cos_phi, sin_phi, aperture

    def farfield_to_back_focal_plane_unit_vectors(self, E: Tuple[np.ndarray], s: Tuple[np.ndarray]):
        Ex_bfp, Ey_bfp = farfield_to_back_focal_plane_unit_vectors(
            *E, *s, self.n_medium, self.n_bfp
        )
        return BackFocalPlaneFields(Ex=Ex_bfp, Ey=Ey_bfp)

    def farfield_to_back_focal_plane_cosines(
        self, E: Tuple[np.ndarray], cos_theta, cos_phi, sin_phi
    ):
        Ex_bfp, Ey_bfp = farfield_to_back_focal_plane_cosines(
            *E, cos_theta, cos_phi, sin_phi, self.n_medium, self.n_bfp
        )
        return BackFocalPlaneFields(Ex=Ex_bfp, Ey=Ey_bfp)


def farfield_to_back_focal_plane_unit_vectors(
    Exff: np.ndarray,
    Eyff: np.ndarray,
    Ezff: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    n_medium: float,
    n_bfp: float,
):
    Exff, Eyff, Ezff, sx, sy, sz = [np.asarray(arr) for arr in (Exff, Eyff, Ezff, sx, sy, sz)]
    if not all([arr.shape == Exff.shape for arr in (Exff, Eyff, Ezff, sx, sy, sz)]):
        raise RuntimeError("All input arrays need to be of the same size")

    cosP, sinP = cosines_from_unit_vectors(sx, sy)

    return farfield_to_back_focal_plane_cosines(
        Exff=Exff,
        Eyff=Eyff,
        Ezff=Ezff,
        cos_theta=sz,
        cos_phi=cosP,
        sin_phi=sinP,
        n_medium=n_medium,
        n_bfp=n_bfp,
    )


def farfield_to_back_focal_plane_cosines(
    Exff: np.ndarray,
    Eyff: np.ndarray,
    Ezff: np.ndarray,
    cos_theta: np.ndarray,
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    n_medium: float,
    n_bfp: float,
):
    Exff, Eyff, Ezff, cos_phi, sin_phi, cos_theta = [
        np.asarray(arr) for arr in (Exff, Eyff, Ezff, cos_phi, sin_phi, cos_theta)
    ]
    Et, Ep, Ex_bfp, Ey_bfp = [np.zeros(Exff.shape, dtype="complex128") for _ in range(4)]
    # Support multiple farfields for same angles
    cos_theta, cos_phi, sin_phi = [
        np.broadcast_to(array, Exff.shape) for array in (cos_theta, cos_phi, sin_phi)
    ]
    sin_theta = ((1 + cos_theta) * (1 - cos_theta)) ** 0.5

    roc = cos_theta > 0  # roc == Region of convergence, avoid division by zero

    Et[roc] = (
        (
            (Exff[roc] * cos_phi[roc] + Eyff[roc] * sin_phi[roc]) * cos_theta[roc]
            - Ezff[roc] * sin_theta[roc]
        )
        * (n_medium / n_bfp) ** 0.5
        / (cos_theta[roc]) ** 0.5
    )

    Ep[roc] = (
        (Eyff[roc] * cos_phi[roc] - Exff[roc] * sin_phi[roc])
        * (n_medium / n_bfp) ** 0.5
        / (cos_theta[roc]) ** 0.5
    )

    Ex_bfp[roc] = Et[roc] * cos_phi[roc] - Ep[roc] * sin_phi[roc]
    Ey_bfp[roc] = Ep[roc] * cos_phi[roc] + Et[roc] * sin_phi[roc]

    return Ex_bfp, Ey_bfp

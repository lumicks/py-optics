import math
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import astuple, dataclass

import numpy as np
from numpy.typing import ArrayLike

from .farfield_data import FarfieldData


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
            Has to be strictly positive and real.
        n_medium : float
            Refractive index of the immersion medium of the objective. Has to be strictly positive,
            real, and larger than the NA.

        Raises
        ------
        ValueError
            Raised if `NA` > `n_medium`, if `n_medium < 0` or `n_bp < 0`, if `focal_length < 0` or
            if `NA < 0`.
        """
        if (
            isinstance(n_medium, complex)
            or isinstance(n_bfp, complex)
            or n_medium <= 0
            or n_bfp <= 0
        ):
            raise ValueError(
                "Only positive and real refractive indices are supported for n_bfp and n_medium"
            )
        if NA > n_medium:
            raise ValueError("The NA of the objective cannot be larger than n_medium")

        if focal_length <= 0:
            raise ValueError("focal_length needs to be strictly positive")

        if NA <= 0 or isinstance(NA, complex):
            raise ValueError("NA needs to be strictly positive and real")

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

        Ex_bfp, Ey_bfp = (
            f_input_field(*astuple(bfp_coords)) if f_input_field is not None else (None, None)
        )

        return bfp_coords, BackFocalPlaneFields(Ex=Ex_bfp, Ey=Ey_bfp)

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
        bfp_sampling_n = bfp_coords.bfp_sampling_n
        sin_theta_x = bfp_coords.x_bfp / self.focal_length
        sin_theta_y = bfp_coords.y_bfp / self.focal_length
        sin_theta = bfp_coords.r_bfp / self.focal_length * bfp_coords.aperture
        aperture = bfp_coords.aperture  # sin_theta <= self.sin_theta_max
        # Calculate properties of the plane waves in the far field
        k = 2 * np.pi * self.n_medium / lambda_vac

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

        kz = k * cos_theta
        kp = k * sin_theta
        kx = -kp * cos_phi
        ky = -kp * sin_phi

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
            kp=kp,
            Einf_theta=Einf_theta,
            Einf_phi=Einf_phi,
            aperture=bfp_coords.aperture,
        )

    def minimal_sampling_order(
        self,
        coordinates: Iterable[ArrayLike],
        lambda_vac: float,
        method: str = "equidistant",
    ) -> int:
        """
        Retrieve the absolute minimal sampling order for sampling the back focal plane, when
        focusing a beam of light. For the only support method "equidistant" (see below), the
        minimally required sampling order is based on the Nyquist sampling theorem.

        Parameters
        ----------
        coordinates : Iterable[ArrayLike]
            The coordinates in the range of interest. First index should iterate over axis:
            `coordinates[0, ...]` for x, `coordinates[1, ...]` for y and `coordinates[2, ...]` for
            z. The units can be anything, as long as the wavelength `lambda_vac` (see below) is
            provided in the same units.
        lambda_vac : float
            Wavelength of the light in vacuum. Has to be in the same units as the coordinates.
        method : str, optional
            Integration method, by default "equidistant". Currently only "equidistant" is supported,
            which means that the back focal plane is sampled with equidistant steps. In the future,
            other methods may be supported. See Notes for an explanation of the methods

        Returns
        -------
        int
            The sampling order N that is the minimum required, for the specific method.

        Raises
        ------
        ValueError
            Raised if an unknown method is passed.
        RuntimeError
            Raised if `len(coordinates)` is not equal to three.

        Notes
        -----
        The currently only supported method of sampling the back focal plane is "equidistant". This
        method samples the back focal plane with equidistant steps. For proper sampling, the number
        of samples should be at least 2 per oscillation, where the frequency of the oscillation
        depends on the maximum magnitude of the coordinates.

        For in-plane vectors: :math:`\\max(|\\Delta k_\\rho \\rho|) < \\pi`, where :math:`\\Delta
        k_\\rho = k_{\\rho, max} / (N - 1)`. :math:`k_{\\rho, max} = k \\sin(\theta)_{max} = k
        NA/n_{medium}`.

        For out-of-plane vectors: :math:`\\max(|\\Delta k_z z|) < \\pi`, where :math:`\\Delta k_z /
        k = \\sqrt{1 - ((N - 2) NA)^2 /(n_{medium} (N - 1))^2} - \\sqrt{1 - (NA /n_{medium})^2}`.

        The function returns :math:`\\max(N_\\rho, N_z)`, the minimal number of samples to fulfill
        the sampling theorem.
        """
        if method != "equidistant":
            raise ValueError(f"Unsupported method: {method}")

        if len(coordinates) != 3:
            raise RuntimeError(
                f"Unexpected length of coordinates: expected 3, got {len(coordinates)}."
            )

        def N_z():
            if max_z == 0.0:  # sampling not affected by z coordinate
                return 2
            # Threshold is sampling criterion + cos(theta_max)
            # == pi / (k*max_z) + (1 - (NA /n_medium))**2
            threshold = (
                lambda_vac / (2 * self.n_medium * max_z) + (1 - self.sin_theta_max**2) ** 0.5
            )
            if threshold > 1:
                return 2
            # Solving for 1-(N-2)**2/(N-1)**2 * (NA/n_medium)**2 < threshold
            # with x = (N-2)/(N-1) yields:
            x = ((1 - threshold**2) / self.sin_theta_max**2) ** 0.5
            # Solve for N in x = (N-2)/(N-1):
            return math.ceil((x - 2) / (x - 1))

        max_x, max_y, max_z = [np.max(np.abs(ax)) for ax in coordinates]

        N_rho = math.ceil(1 + 2 * self.NA * math.hypot(max_x, max_y) / lambda_vac)
        return max(N_rho, N_z())

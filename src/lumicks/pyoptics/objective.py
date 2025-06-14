import math
from collections import namedtuple
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from .farfield_data import FarfieldData
from .mathutils.integration.disk import get_integration_locations


@dataclass
class BackFocalPlaneCoordinates:
    """
    Class to store data necessary for doing calculations with the back focal
    plane
    """

    weights: np.ndarray
    x_bfp: np.ndarray
    y_bfp: np.ndarray
    r_bfp: np.ndarray


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
    def r_bfp_max(self) -> float:
        """Radius of the back focal plane"""
        return self.sin_theta_max * self.focal_length

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

    def sample_back_focal_plane(self, f_input_field, order: int, method: str | None = None):
        """Sample `f_input_field` and return the coordinates in a BackFocalPlaneCoordinates object
        and the fields as a named tuple BackFocalPlaneFields(Ex, Ey).

        Parameters
        ----------
        f_input_field : Callable
            Function with signature `f(coordinates: BackFocalPlaneCoordinates, objective:
            Objective)`. The function should take the coordinates and return the electric field at
            that location on the back focal plane. The returned value should be a tuple, where the
            first entry is the x-polarized field, and the second entry is the y-polarized field.
        order : int
            Integration order for the chosen method (see below).
        method : str | None, optional
            Method of integration, by default None. Needs to be explicitly specified. Options are
            "equidistant" and "peirce".
        """

        def _equidistant_coords():
            sin_theta = self.sine_theta_range(order)
            x_bfp, y_bfp = np.meshgrid(sin_theta, sin_theta, indexing="ij")

            sin_theta = np.hypot(x_bfp, y_bfp)
            x_bfp *= self.focal_length
            y_bfp *= self.focal_length

            r_bfp = sin_theta * self.focal_length
            dx = self.r_bfp_max / (order - 1)
            weights = (sin_theta <= self.sin_theta_max).astype(float) * dx**2
            return BackFocalPlaneCoordinates(
                weights=weights,
                x_bfp=x_bfp,
                y_bfp=y_bfp,
                r_bfp=r_bfp,
            )

        def _disk_coords(method: str):
            x_bfp, y_bfp, w = get_integration_locations(order, method=method)
            w *= self.r_bfp_max**2
            x_bfp *= self.r_bfp_max
            y_bfp *= self.r_bfp_max
            r_bfp = np.hypot(x_bfp, y_bfp)
            return BackFocalPlaneCoordinates(
                weights=w,
                x_bfp=x_bfp,
                y_bfp=y_bfp,
                r_bfp=r_bfp,
            )

        if method == "equidistant":
            bfp_coords = _equidistant_coords()
        elif method in ["peirce", "lether", "takaki"]:
            bfp_coords = _disk_coords(method)
        else:
            raise ValueError(f"Sampling method {method} is not supported.")

        if f_input_field is not None:
            Ex_bfp, Ey_bfp = f_input_field(bfp_coords, self)
            if Ex_bfp is None and Ey_bfp is None:
                raise RuntimeError("Either an x-polarized or a y-polarized input field is required")
        else:
            Ex_bfp, Ey_bfp = None, None
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
        sin_theta_x = bfp_coords.x_bfp / self.focal_length
        sin_theta_y = bfp_coords.y_bfp / self.focal_length
        sin_theta = bfp_coords.r_bfp / self.focal_length * (bfp_coords.weights > 0.0)
        aperture = bfp_coords.weights > 0.0
        # Calculate properties of the plane waves in the far field
        k = 2 * np.pi * self.n_medium / lambda_vac

        cos_theta = np.ones(sin_theta.shape)
        cos_theta[aperture] = ((1 + sin_theta[aperture]) * (1 - sin_theta[aperture])) ** 0.5

        cos_phi = np.ones_like(sin_theta)
        sin_phi = np.zeros_like(sin_theta)
        region = sin_theta > 0 & aperture

        cos_phi[region] = sin_theta_x[region] / sin_theta[region]

        cos_phi[np.logical_and(bfp_coords.x_bfp == 0.0, bfp_coords.y_bfp == 0.0)] = 1
        sin_phi[region] = sin_theta_y[region] / sin_theta[region]
        sin_phi[np.logical_and(bfp_coords.x_bfp == 0.0, bfp_coords.y_bfp == 0.0)] = 0
        sin_phi[np.logical_not(aperture)] = 0
        cos_phi[np.logical_not(aperture)] = 1

        kz = k * cos_theta
        kp = k * sin_theta
        kx = -kp * cos_phi
        ky = -kp * sin_phi

        # Transform the input wavefront to a spherical one, after refracting on the Gaussian
        # reference sphere [2], Ch. 3. The field magnitude changes because of the different media,
        # and because of the angle (preservation of power in a beamlet).
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
        weights = bfp_coords.weights * (self.sin_theta_max * k / self.r_bfp_max) ** 2
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
            weights=weights,
        )

    def minimal_integration_order(
        self,
        coordinates: Collection,
        lambda_vac: float,
        method: str,
    ) -> int:
        """
        Retrieve the absolute minimal integration order for sampling the back focal plane, when
        focusing a beam of light.

        Parameters
        ----------
        coordinates : Collection
            The coordinates in the range of interest. First index should iterate over axis:
            `coordinates[0, ...]` for x, `coordinates[1, ...]` for y and `coordinates[2, ...]` for
            z. The units can be anything, as long as the wavelength `lambda_vac` (see below) is
            provided in the same units.
        lambda_vac : float
            Wavelength of the light in vacuum. Has to be in the same units as the coordinates.
        method : str
            Integration method. Supported are "equidistant" and "peirce". See Notes for an
            explanation of the methods.

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
        Method "equidistant": this method samples the back focal plane with equidistant steps. For
        proper sampling, the number of samples should be at least 2 per oscillation, where the
        frequency of the oscillation depends on the maximum magnitude of the coordinates. It's
        mostly useful for FFT/CZT-based point spread function calculation methods, such as
        `lumicks.pyoptics.psf.fast_psf` and `lumicks.pyoptics.psf.fast_gauss`.

        For in-plane vectors: :math:`\\Delta k_{//} \\max(|x|, |y|) < \\pi`, where :math:`\\Delta
        k_{//} = k NA / (n_\\mathit{medium} (N - 1)) = k \\sin(\\theta_\\mathit{max}) / (N - 1)`.

        For out-of-plane vectors: :math:`\\max(\\Delta k_z) |z|| < \\pi`, where :math:`\\max(\\Delta
        k_z) / k = \\sqrt{1 - ((N - 2) NA)^2 /(n_{medium} (N - 1))^2} - \\sqrt{1 - (NA
        /n_{medium})^2}`.

        The function returns :math:`\\max(N_{//}, N_z)`, the minimal number of samples to fulfill
        the sampling theorem.

        Method "peirce": this method samples the back focal plane with a radial scheme based on
        Gauss-Legendre-style integration. This method typically converges faster than the
        "equidistant" method. See[2]_.

        In order to determine the minimal integration order, integration over the back focal plane
        is divided between coordinates along the optical axis (z) and orthogonal to the optical axis
        (x and y). For the xy-plane, the function assumes the integration of a (1D) sinusoidal
        function, and uses an analytical expression for the error bound[1]_. The function finds the
        order that ensures that the error bound is 1e-9 times the maximum error bound. For the
        coordinates along the optical axis, a toy model function with a :math: `r \\cos(k z \\sqrt{1
        - (NA /n_{medium})^2 r})`-dependency is integrated, until the relative difference between
        successive integration results is less than 1e-9. The function returns :math:`\\max(N_{xy},
        N_z)`

        .. [1] Abramowitz & Stegun, 25.4.30
        .. [2] W. Peirce, "Numerical Integration over the Planar Annulus", Journal of the Society
               for Industrial and Applied Mathematics, 1957

        """
        if method not in ["equidistant", "peirce"]:
            raise ValueError(f"Unsupported method: {method}")

        if len(coordinates) != 3:
            raise RuntimeError(
                f"Unexpected length of coordinates: expected 3, got {len(coordinates)}."
            )

        if method == "equidistant":
            return self._minimal_integration_order_equidistant(coordinates, lambda_vac)
        if method == "peirce":
            return self._minimal_integration_order_peirce(coordinates, lambda_vac)

    def _minimal_integration_order_equidistant(self, coordinates, lambda_vac):

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

        N_parallel = math.ceil(1 + 2 * self.NA * max(max_x, max_y) / lambda_vac)
        return max(N_parallel, N_z())

    def _minimal_integration_order_peirce(
        self, coordinates: Collection, lambda_vac: float, max_iterations: int = 400
    ):
        def _min_order_xy(max_iterations=max_iterations):
            def cost(a, b, n, x):
                # Cost function based on A&S 25.4.30. Uses Stirling's approximation to n!, and works in
                # logarithmic space. The 2n-th derivative of f is replaced by x^(2n), which is the
                # resulting prefactor when differentiating a sinusoid of the type sin(k x) 2n times.
                return (
                    (2 * n + 1) * math.log(b - a)
                    + 4 * (n * math.log(n) - n)
                    + 2 * n * math.log(x)
                    - (math.log(2 * n + 1) + 3 * (2 * n * math.log(2 * n) - 2 * n))
                )

            n = 1
            max_cost = cost(0, 2 * np.pi * self.NA / lambda_vac, n, max_xy)
            while (  # First get to the maximum cost (first increases, then decreases)
                max_cost < (new_cost := cost(0, 2 * np.pi * self.NA / lambda_vac, n + 1, max_xy))
                and max_iterations > 0
            ):
                n += 1
                max_iterations -= 1
                max_cost = new_cost

            max_cost = max(max_cost, cost(0, 2 * np.pi * self.NA / lambda_vac, n, max_xy))
            while (  # Now iterate until the cost drops sufficiently, or we're out of iterations
                cost(0, 2 * np.pi * self.NA / lambda_vac, n + 1, max_xy) - max_cost > np.log(1e-9)
                and max_iterations > 0
            ):
                n += 1
                max_iterations -= 1
            n_xy = n
            return n_xy

        def _min_order_z(max_iterations=max_iterations):
            from scipy.special import roots_legendre

            def g(x):
                # Toy model function for the complex exponential e^(1j k_z z). Note that k_z is the
                # independent variable.
                return np.cos(
                    2 * np.pi * (1 - self.NA / self.n_medium * x**2) ** 0.5 * max_z / lambda_vac
                )

            def integral_g(n):
                xi, w = roots_legendre(n)
                w /= 2
                xi = (xi * 0.5 + 0.5) ** 0.5
                return np.sum(g(xi) * w)

            y = integral_g(max(1, n_xy - 1))
            for n in np.arange(max(2, n_xy), max_iterations + 1):
                y_new = integral_g(n)
                if np.abs((y_new - y) / y_new) < 1e-9:
                    n_z = n
                    break
                else:
                    y = y_new
            return n_z

        max_x, max_y, max_z = [np.max(np.abs(ax)) for ax in coordinates]
        n_z = n_xy = 1

        if (max_xy := math.hypot(max_x, max_y)) > 0:
            n_xy = _min_order_xy()

        if max_z > 0:
            n_z = _min_order_z()

        return max(n_xy, n_z)

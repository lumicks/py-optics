import numpy as np
from dataclasses import dataclass, astuple
from collections import namedtuple
from typing import Union
from .farfield_data import FarfieldData
from .mathutils import czt


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
    Class to describe the essential properties of an objective
    """

    def __init__(self, NA: float, focal_length: float, n_bfp: float, n_medium: float):
        if NA > n_medium:
            raise ValueError("The NA of the objective cannot be larger than n_medium")

        if n_medium <= 0 or n_bfp <= 0:
            raise ValueError(
                "Only positive refractive indices are supported for n_bfp and n_medium"
            )

        if focal_length <= 0:
            raise ValueError("focal_length needs to be positive")

        if NA <= 0:
            raise ValueError("NA needs to be positive")

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

        Ex_bfp, Ey_bfp = f_input_field(*astuple(bfp_coords))

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
        sin_theta = self.sine_theta_range(bfp_sampling_n)
        sin_theta_x, sin_theta_y = np.meshgrid(sin_theta, sin_theta, indexing="ij")
        sin_theta = np.hypot(sin_theta_x, sin_theta_y)

        # Calculate properties of the plane waves in the far field
        k = 2 * np.pi * self.n_medium / lambda_vac

        cos_theta = np.ones(sin_theta.shape)
        cos_theta[bfp_coords.aperture] = (
            (1 + sin_theta[bfp_coords.aperture]) * (1 - sin_theta[bfp_coords.aperture])
        ) ** 0.5

        cos_phi = np.ones_like(sin_theta)
        sin_phi = np.zeros_like(sin_theta)
        region = sin_theta > 0

        cos_phi[region] = sin_theta_x[region] / sin_theta[region]

        cos_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 1
        sin_phi[region] = sin_theta_y[region] / sin_theta[region]
        sin_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 0
        sin_phi[np.logical_not(bfp_coords.aperture)] = 0
        cos_phi[np.logical_not(bfp_coords.aperture)] = 1

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
                E[np.logical_not(bfp_coords.aperture)] = 0
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

    def focus(
        self,
        f_input_field,
        lambda_vac: float,
        x_range: Union[float, tuple[float, float]],
        numpoints_x: int,
        y_range: Union[float, tuple[float, float]],
        numpoints_y: int,
        z: np.array,
        bfp_sampling_n=125,
        return_grid=False,
        bias_correction=True,
    ):
        """Calculate the 3-dimensional, vectorial Point Spread Function of an
        arbitrary input field, using the angular spectrum of plane waves method, see [1]_, chapter
        3. This function uses the chirp-z transform for speedy evaluation of the fields in the focus
        [2]_.

        This function correctly incorporates the polarized nature of light in a focus. In other
        words, the polarization state at the focus includes electric fields in the x, y, and z
        directions. The input can be any arbitrary polarization in the XY plane.

        Parameters
        ----------
        f_input_field : callable
            Function with signature `f(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n)`, where
            `x_bfp` is a grid of x locations in the back focal plane, determined by the focal length
            and NA of the objective. The corresponding grid for y is `y_bfp`, and `r_bfp` is the
            radial distance from the center of the back focal plane. The float `r_max` is the
            largest distance that falls inside the NA, but `r_bfp` will contain larger numbers as
            the back focal plane is sampled with a square grid. The number of samples from the
            center to the edge of the NA in horizontal or vertical direction is `bfp_sampling_n`.
            This number is forwarded to the callable for convenience. The function must return a
            tuple `(E_bfp_x, E_bfp_y)`, which are the electric fields in the x- and y- direction,
            respectively, at the sample locations in the back focal plane. In other words, `E_bfp_x`
            describes the electric field of the input beam which is polarized along the x-axis.
            Similarly, `E_bfp_y` describes the y-polarized part of the input beam. The fields may be
            complex, so a phase difference between x and y is possible. If only one polarization is
            used, the other return value must be `None`, e.g., y polarization would return `(None,
            E_bfp_y)`. The fields are post-processed such that any part that falls outside of the NA
            is set to zero.
        lambda_vac : float
            Wavelength of the light [m]
        x_range : Union[float, tuple(float, float)]
            Size of the calculation range along x, in meters. If the range is a single float, it is
            centered around zero. The algorithm will calculate at x locations
            [-x_range/2..x_range/2]. Otherwise, it will calculate at locations from
            [x_range[0]..x_range[1]]
        numpoints_x : int
            Number of points to calculate along the x dimension. Must be >= 1
        y_range : Union[float, tuple(float, float)]
            Same as x, but along y [m]
        numpoints_y : int
            Same as x, but for y
        z : Union[np.array, float]
            Numpy array of locations along z, where to calculate the fields. Can be a single number
            as well [m]
        bfp_sampling_n : int, optional
            Number of discrete steps with which the back focal plane is sampled, from the center to
            the edge. The total number of plane waves scales with the square of bfp_sampling_n
            (default = 125)
        return_grid : bool, optional
            Return the sampling grid (default = False)

        Returns
        -------
        Ex : ndarray
            the electric field along x, as a function of (x, y, z) [V/m]
        Ey : ndarray
            the electric field along y, as a function of (x, y, z) [V/m]
        Ez : ndarray
            the electric field along z, as a function of (x, y, z) [V/m]

        If return_grid is True, then also return the sampling grid X, Y and Z.

        All results are returned with the minimum number of dimensions required to store the
        results, i.e., sampling along the XZ plane will return the fields as Ex(x,z), Ey(x,z),
        Ez(x,z)

        Raises
        ------


        References
        ----------
        .. [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.). Cambridge:
            Cambridge University Press. doi:10.1017/CBO9780511794193
        .. [2] Marcel Leutenegger, Ramachandra Rao, Rainer A. Leitgeb, and Theo Lasser, "Fast focus
            field calculations," Opt. Express 14, 11277-11291 (2006)
        """
        if numpoints_x < 1:
            raise ValueError("numpoints_x needs to be >= 1")
        if numpoints_y < 1:
            raise ValueError("numpoints_y needs to be >= 1")
        z = np.atleast_1d(z)
        x_range = np.asarray(x_range, dtype="float")
        y_range = np.asarray(y_range, dtype="float")
        if x_range.size == 1:
            if numpoints_x > 1:
                raise ValueError("x_range needs to be a tuple (xmin, xmax) for numpoints_x > 1")
            x_center = x_range
            x_range = 0.0
        elif x_range.size == 2:
            x_center = np.mean(x_range)
            x_range = np.abs(np.diff(x_range))
        else:
            raise ValueError(f"Unexpected size of {x_range.size} elements for x_range")
        if y_range.size == 1:
            if numpoints_y > 1:
                raise ValueError("y_range needs to be a tuple (ymin, ymax) for numpoints_x > 1")
            y_center = y_range
            y_range = 0.0
        elif y_range.size == 2:
            y_center = np.mean(y_range)
            y_range = np.abs(np.diff(y_range))
        else:
            raise ValueError(f"Unexpected size of {y_range.size} elements for y_range")

        k = 2 * np.pi * self.n_medium / lambda_vac
        ks = k * self.NA / self.n_medium
        npupilsamples = 2 * bfp_sampling_n - 1

        dk = ks / (bfp_sampling_n - 1)

        bfp_coords, bfp_fields = self.sample_back_focal_plane(
            f_input_field=f_input_field,
            bfp_sampling_n=bfp_sampling_n,
        )
        if bfp_fields.Ex is None and bfp_fields.Ey is None:
            raise RuntimeError("Either an x-polarized or a y-polarized input field is required")

        far_field = self.back_focal_plane_to_farfield(
            bfp_coords=bfp_coords, bfp_fields=bfp_fields, lambda_vac=lambda_vac
        )
        # Multiply the transformed field by kz^-1 for the integral
        Ex_inf, Ey_inf, Ez_inf = [field / far_field.kz for field in far_field.transform_to_xyz()]
        # Make kz 3D - np.atleast_3d() prepends a dimension, and that is not what we need
        kz = np.reshape(far_field.kz, (npupilsamples, npupilsamples, 1))

        Z = np.tile(z, ((2 * bfp_sampling_n - 1), (2 * bfp_sampling_n - 1), 1))
        Exp = np.exp(1j * kz * Z)

        Ex_inf, Ey_inf, Ez_inf = [
            np.tile(np.reshape(E, (npupilsamples, npupilsamples, 1)), (1, 1, z.shape[0])) * Exp
            for E in (Ex_inf, Ey_inf, Ez_inf)
        ]

        prefactor = (
            -1j * self.focal_length * np.exp(-1j * k * self.focal_length) * dk**2 / (2 * np.pi)
        )
        if bias_correction:
            prefactor *= ks**2 * np.pi / (dk**2 * np.count_nonzero(bfp_coords.aperture))

        # Set up the factors for the chirp z transform
        ax = np.exp(-1j * dk * (0.5 * x_range - x_center))
        wx = np.exp(-1j * dk * x_range / (numpoints_x - 1)) if numpoints_x > 1 else 1.0
        ay = np.exp(-1j * dk * (0.5 * y_range - y_center))
        wy = np.exp(-1j * dk * y_range / (numpoints_y - 1)) if numpoints_y > 1 else 1.0

        # The chirp z transform assumes data starting at x[0], but our aperture is
        # symmetric around point (0,0). Therefore, fix the phases after the
        # transform such that the real and imaginary parts of the fields are what
        # they need to be
        phase_fix_x = np.reshape(
            (ax * wx ** -(np.arange(numpoints_x))) ** (bfp_sampling_n - 1), (numpoints_x, 1, 1)
        )
        phase_fix_step1 = np.tile(phase_fix_x, (1, npupilsamples, Z.shape[2]))

        phase_fix_y = np.reshape(
            (ay * wy ** -(np.arange(numpoints_y))) ** (bfp_sampling_n - 1), (numpoints_y, 1, 1)
        )

        phase_fix_step2 = np.tile(phase_fix_y, (1, numpoints_x, Z.shape[2]))

        # We break the czt into two steps, as there is an overlap in processing that
        # needs to be done for every polarization. Therefore we can save a bit of
        # overhead by storing the results that can be reused.
        precalc_step1 = czt.init_czt(Ex_inf, numpoints_x, wx, ax)
        Ex = np.transpose(czt.exec_czt(Ex_inf, precalc_step1) * phase_fix_step1, (1, 0, 2))
        precalc_step2 = czt.init_czt(Ex, numpoints_y, wy, ay)
        Ex = np.transpose(czt.exec_czt(Ex, precalc_step2) * phase_fix_step2, (1, 0, 2))
        Ex *= prefactor
        E = []
        for E_inf in (Ey_inf, Ez_inf):
            field = np.transpose(czt.exec_czt(E_inf, precalc_step1) * phase_fix_step1, (1, 0, 2))
            field = np.transpose(czt.exec_czt(field, precalc_step2) * phase_fix_step2, (1, 0, 2))
            E.append(field * prefactor)

        Ey, Ez = E

        retval = (np.squeeze(Ex), np.squeeze(Ey), np.squeeze(Ez))

        if return_grid:
            xrange_v = np.linspace(-0.5 * x_range + x_center, 0.5 * x_range + x_center, numpoints_x)
            yrange_v = np.linspace(-0.5 * y_range + y_center, 0.5 * y_range + y_center, numpoints_y)
            X, Y, Z = np.meshgrid(xrange_v, yrange_v, np.squeeze(z), indexing="ij")
            retval += (np.squeeze(X), np.squeeze(Y), np.squeeze(Z))

        return retval

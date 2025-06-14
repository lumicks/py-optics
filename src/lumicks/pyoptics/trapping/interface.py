import logging
from typing import Optional, Tuple

import numpy as np
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0
from scipy.constants import speed_of_light as _C

from ..mathutils.integration.sphere import get_integration_locations, get_nearest_order
from ..objective import BackFocalPlaneCoordinates, Objective
from .bead import Bead, _determine_integration_order
from .focused_field_calculation import focus_field_factory
from .local_coordinates import LocalBeadCoordinates
from .plane_wave_field_calculation import plane_wave_field_factory


def fields_focus_gaussian(
    beam_power: float,
    filling_factor,
    objective: Objective,
    bead: Bead,
    bead_center=(0, 0, 0),
    x=0,
    y=0,
    z=0,
    bfp_sampling_n=31,
    num_orders=None,
    return_grid=False,
    total_field=True,
    magnetic_field=False,
    verbose=False,
    grid=True,
):
    """
    Calculate the three-dimensional electromagnetic field of a bead the focus of a of a Gaussian
    beam, using the angular spectrum of plane waves and Mie theory [1]_[2]_.

    This function correctly incorporates the polarized nature of light in a focus. In other words,
    the polarization state at the focus includes electric fields in the x, y, and z directions. The
    input is taken to be polarized along the x direction.

    Parameters
    ----------
    beam_power : float
        Power of the laser beam before entering the objective, in Watt.
    filling_factor : float
        Filling factor of the Gaussian beam over the aperture, defined as w0/R. Here, w0 is the
        waist of the Gaussian beam and R is the radius of the aperture. Range 0...Inf
    objective : Objective
        Instance of the Objective class.
    bead: Bead
        Instance of the Bead class
    x : np.ndarray
        array of x locations for evaluation, in meters
    y : np.ndarray
        array of y locations for evaluation, in meters
    z : np.ndarray
        array of z locations for evaluation, in meters
    bead_center : tuple
        A tuple of three floating point numbers determining the x, y and z position of the bead
        center in 3D space, in meters
    bfp_sampling_n : (Default value = 31) Number of discrete steps with
        which the back focal plane is sampled, from the center to the edge. The total number of
        plane waves scales with the square of bfp_sampling_n
    num_orders : number of order that should be included in the
        calculation the Mie solution. If it is `None` (default), the code will use the
        `number_of_orders()` method to calculate a sufficient number.
    return_grid: bool, optional
        return the sampling grid in the matrices X, Y and Z. Default value = False
    total_field : bool, optional
        If True, return the total field of incident and scattered electromagnetic field (default).
        If False, then only return the scattered field outside the bead. Inside the bead, the full
        field is always returned.
    magnetic_field : bool, optional
        If True, return the magnetic fields as well. If false (default), do not return the magnetic
        fields.
    verbose : bool, optional
        If True, print statements on the progress of the calculation. Default is False
    grid: bool, optional
        If True (default), interpret the vectors or scalars x, y and z as the input for the
        numpy.meshgrid function, and calculate the fields at the locations that are the result of
        the numpy.meshgrid output. If False, interpret the x, y and z vectors as the exact locations
        where the field needs to be evaluated. In that case, all vectors need to be of the same
        length.

    Returns
    -------
    Ex : np.ndarray
        The electric field along x, as a function of (x, y, z)
    Ey : np.ndarray
        The electric field along y, as a function of (x, y, z)
    Ez : np.ndarray
        The electric field along z, as a function of (x, y, z)
    Hx : np.ndarray
        The magnetic field along x, as a function of (x, y, z). Only returned when magnetic_field is
        True
    Hy : np.ndarray
        The magnetic field along y, as a function of (x, y, z). Only returned when magnetic_field is
        True
    Hz : np.ndarray
        The magnetic field along z, as a function of (x, y, z). Only returned when magnetic_field is
        True
    X : np.ndarray
        x coordinates of the sampling grid
    Y : np.ndarray
        y coordinates of the sampling grid
    Z : np.ndarray
        z coordinates of the sampling grid These values are only returned if return_grid is True

    ..  [1] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
            Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193
    ..  [2] Craig F. Bohren & Donald R. Huffman (1983). Absorption and Scattering of
            Light by Small Particles. WILEY‐VCH Verlag GmbH & Co. KGaA.
            doi:10.1002/9783527618156
    """
    w0 = filling_factor * objective.focal_length * objective.NA / bead.n_medium  # [m]
    I0 = 2 * beam_power / (np.pi * w0**2)  # [W/m^2]
    E0 = (I0 * 2 / (EPS0 * _C * objective.n_bfp)) ** 0.5  # [V/m]

    def gaussian_beam(coordinates: BackFocalPlaneCoordinates, objective: Objective):
        Ex = np.exp(-(coordinates.x_bfp**2 + coordinates.y_bfp**2) / w0**2) * E0
        return (Ex, None)

    return fields_focus(
        gaussian_beam,
        objective,
        bead,
        x=x,
        y=y,
        z=z,
        bead_center=bead_center,
        bfp_sampling_n=bfp_sampling_n,
        num_orders=num_orders,
        return_grid=return_grid,
        total_field=total_field,
        magnetic_field=magnetic_field,
        verbose=verbose,
        grid=grid,
    )


def fields_focus(
    f_input_field,
    objective: Objective,
    bead: Bead,
    bead_center=(0.0, 0.0, 0.0),
    x=0.0,
    y=0.0,
    z=0.0,
    bfp_sampling_n=31,
    num_orders=None,
    return_grid=False,
    total_field=True,
    magnetic_field=False,
    verbose=False,
    grid=True,
):
    """
    Calculate the three-dimensional electromagnetic field of a bead in the focus of an arbitrary
    input beam, going through an objective with a certain NA and focal length. Implemented with the
    angular spectrum of plane waves and Mie theory.

    This function correctly incorporates the polarized nature of light in a focus. In other words,
    the polarization state at the focus includes electric fields in the x, y, and z directions. The
    input can be a combination of x- and y-polarized light of complex amplitudes.

    Parameters
    ----------
    f_input_field : callable
        function with signature `f(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n)`, where
        `x_bfp` is a grid of x locations in the back focal plane, determined by the focal length and
        NA of the objective. `y_bfp` is the corresponding grid of y locations, and `r_bfp` is the
        radial distance from the center of the back focal plane. r_max is the largest distance that
        falls inside the NA, but r_bfp will contain larger numbers as the back focal plane is
        sampled with a square grid. The function must return a tuple (Ex, Ey), which are the
        electric fields in the x- and y- direction, respectively, at the sample locations in the
        back focal plane. The fields may be complex, so a phase difference between x and y is
        possible. If only one polarization is used, the other return value must be None, e.g., y
        polarization would return (None, Ey). The fields are post-processed such that any part that
        falls outside of the NA is set to zero. This region is indicated by the variable `aperture`,
        which has the same shape as `x_bfp`, `y_bfp` and `r_bfp`, and for every location indicates
        whether it is inside the NA of the objective with `True` or outside the NA with `False`. The
        integer `bfp_sampling_n` is the number of samples of the back focal plane from the center to
        the edge of the NA, and is given for convenience or potential caching. This will be the
        number as passed below to `fields_focus()`
    objective : Objective
        Instance of the Objective class
    bead : Bead
        Instance of the Bead class
    x : np.ndarray
        Array of x locations for evaluation, in meters
    y : np.ndarray
        Array of y locations for evaluation, in meters
    z : np.ndarray
        Array of z locations for evaluation, in meters
    bead_center : Tuple[float, float, float]
        Tuple of three floating point numbers determining the x, y and z position of the bead center
        in 3D space, in meters
    bfp_sampling_n : int
        Number of discrete steps with which the back focal plane is sampled, from the center to the
        edge. The total number of plane waves scales with the square of bfp_sampling_n, by default
        31.
    num_orders: int
        Number of orders that should be included in the calculation the Mie solution. If it is None
        (default), the code will use the Bead.number_of_orders() method to calculate a sufficient
        number.
    return_grid : bool
        Return the sampling grid in the matrices X, Y and Z, by default False
    total_field : bool
        If True, return the total field of incident and scattered electromagnetic field (default).
        If False, then only return the scattered field outside the bead. Inside the bead, the full
        field is always returned.
    magnetic_field: bool
        If True, return the magnetic fields as well. If false (default), do not return the magnetic
        fields.
    verbose: bool
        If True, print statements on the progress of the calculation. Default is False
    grid: bool
        If True (default), interpret the vectors or scalars x, y and z as the input for the
        numpy.meshgrid function, and calculate the fields at the locations that are the result of
        the numpy.meshgrid output. If False, interpret the x, y and z vectors as the exact locations
        where the field needs to be evaluated. In that case, all vectors need to be of the same
        length.

    Raises
    ------
    ValueError: raised when the immersion medium of the bead does not match the medium of the
    objective.

    Returns
    -------
    Ex : np.ndarray
        The electric field along x, as a function of (x, y, z)
    Ey : np.ndarray
        The electric field along y, as a function of (x, y, z)
    Ez : np.ndarray
        The electric field along z, as a function of (x, y, z)
    Hx : np.ndarray
        The magnetic field along x, as a function of (x, y, z)
    Hy : np.ndarray
        The magnetic field along y, as a function of (x, y, z)
    Hz : np.ndarray
        The magnetic field along z, as a function of (x, y, z) These values are only returned when
        magnetic_field is True
    X : np.ndarray
        x coordinates of the sampling grid
    Y : np.ndarray
        y coordinates of the sampling grid
    Z : np.ndarray
        z coordinates of the sampling grid. These values are only returned if return_grid is True
    """
    if bead.n_medium != objective.n_medium:
        raise ValueError("The immersion medium of the bead and the objective have to be the same")

    loglevel = logging.getLogger().getEffectiveLevel()
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Enforce floats to ensure Numba has all floats when doing @ / np.matmul
    x, y, z = [np.atleast_1d(coord).astype(np.float64) for coord in (x, y, z)]
    # TODO: warning criteria for undersampling/aliasing
    # M = int(np.max((31, 2 * NA**2 * np.max(np.abs(z)) /
    #        (np.sqrt(self.n_medium**2 - NA**2) * self.lambda_vac))))
    # if M > bfp_sampling_n:
    #    print('bfp_sampling_n lower than recommendation for convergence')

    n_orders = bead.number_of_orders if num_orders is None else max(int(num_orders), 1)
    local_coordinates = LocalBeadCoordinates(x, y, z, bead.bead_diameter, bead_center, grid=grid)

    logging.info("Calculating auxiliary data for external fields")
    field_fun = focus_field_factory(
        objective=objective,
        bead=bead,
        n_orders=n_orders,
        bfp_sampling_n=bfp_sampling_n,
        f_input_field=f_input_field,
        local_coordinates=local_coordinates,
        internal=False,
    )
    logging.info("Calculating external fields")
    external_fields = field_fun(bead_center, True, magnetic_field, total_field)

    logging.info("Calculating auxiliary data for internal fields")
    field_fun = focus_field_factory(
        objective=objective,
        bead=bead,
        n_orders=n_orders,
        bfp_sampling_n=bfp_sampling_n,
        f_input_field=f_input_field,
        local_coordinates=local_coordinates,
        internal=True,
    )

    logging.info("Calculating internal fields")
    internal_fields = field_fun(bead_center, True, magnetic_field)

    for external, internal in zip(external_fields, internal_fields):
        external += internal
    ret = external_fields

    logging.getLogger().setLevel(loglevel)

    if return_grid:
        grid = np.meshgrid(x, y, z, indexing="ij")
        X, Y, Z = (np.squeeze(axis) for axis in grid)
        ret += (X, Y, Z)

    return ret


def fields_plane_wave(
    bead: Bead,
    x,
    y,
    z,
    theta=0,
    phi=0,
    polarization=(1, 0),
    num_orders=None,
    return_grid=False,
    total_field=True,
    magnetic_field=False,
    verbose=False,
    grid=True,
):
    """
    Calculate the electromagnetic field of a bead, subject to excitation
    by a plane wave. The plane wave can be at an angle theta and phi, and
    have a polarization state that is the combination of the (complex)
    amplitudes of a theta-polarization state and phi-polarized state. If
    theta == 0 and phi == 0, the theta polarization points along +x axis and
    the wave travels into the +z direction.

    Parameters
    ----------
    bead: instance of the Bead class
    x : array of x locations for evaluation, in meters
    y : array of y locations for evaluation, in meters
    z : array of z locations for evaluation, in meters
    theta : angle with the negative optical axis (-z)
    phi : angle with the positive x axis
    num_orders : number of order that should be included in the calculation
            the Mie solution. If it is None (default), the code will use the
            number_of_orders() method to calculate a sufficient number.
    return_grid : (Default value = False) return the sampling grid in the
        matrices X, Y and Z
    total_field : If True, return the total field of incident and scattered
        electromagnetic field (default). If False, then only return the
        scattered field outside the bead. Inside the bead, the full field is
        always returned.
    magnetic_field : If True, return the magnetic fields as well. If false
        (default), do not return the magnetic fields.
    verbose : If True, print statements on the progress of the calculation.
        Default is False
    grid : If True (default), interpret the vectors or scalars x, y and z as
        the input for the numpy.meshgrid function, and calculate the fields
        at the locations that are the result of the numpy.meshgrid output.
        If False, interpret the x, y and z vectors as the exact locations
        where the field needs to be evaluated. In that case, all vectors
        need to be of the same length.

    Returns
    -------
    Ex : the electric field along x, as a function of (x, y, z)
    Ey : the electric field along y, as a function of (x, y, z)
    Ez : the electric field along z, as a function of (x, y, z)
    Hx : the magnetic field along x, as a function of (x, y, z)
    Hy : the magnetic field along y, as a function of (x, y, z)
    Hz : the magnetic field along z, as a function of (x, y, z)
        These values are only returned when magnetic_field is True
    X : x coordinates of the sampling grid
    Y : y coordinates of the sampling grid
    Z : z coordinates of the sampling grid
        These values are only returned if return_grid is True
    """
    loglevel = logging.getLogger().getEffectiveLevel()
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Enforce floats to ensure Numba gets the same type in @ / np.matmul
    x, y, z = [np.atleast_1d(c).astype(np.float64) for c in (x, y, z)]

    n_orders = bead.number_of_orders if num_orders is None else max(int(num_orders), 1)
    local_coordinates = LocalBeadCoordinates(x, y, z, bead.bead_diameter, grid=grid)

    logging.info("Calculating auxiliary data for external fields")
    field_fun = plane_wave_field_factory(
        bead=bead,
        n_orders=n_orders,
        theta=theta,
        phi=phi,
        local_coordinates=local_coordinates,
        internal=False,
    )
    logging.info("Calculating external fields")
    external_fields = field_fun(polarization, True, magnetic_field, total_field)

    logging.info("Calculating auxiliary data for internal fields")
    field_fun = plane_wave_field_factory(
        bead=bead,
        n_orders=n_orders,
        theta=theta,
        phi=phi,
        local_coordinates=local_coordinates,
        internal=True,
    )
    logging.info("Calculating internal fields")
    internal_fields = field_fun(polarization, True, magnetic_field)

    for external, internal in zip(external_fields, internal_fields):
        external += internal
    ret = external_fields

    logging.getLogger().setLevel(loglevel)

    if return_grid:
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        X = np.squeeze(X)
        Y = np.squeeze(Y)
        Z = np.squeeze(Z)
        ret += (X, Y, Z)

    return ret


def force_factory(
    f_input_field,
    objective: Objective,
    bead: Bead,
    bfp_sampling_n: int = 31,
    num_orders: Optional[int] = None,
    integration_order: Optional[int] = None,
    method: str = "lebedev-laikov",
):
    """Create and return a function suitable to calculate the force on a bead. Items that can be
    precalculated are stored for rapid subsequent calculations of the force on the bead for
    different positions.

    Parameters
    ----------
    f_input_field : callable
        A callable with the signature `f(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n)`,
        where `x_bfp` is a grid of x locations in the back focal plane, determined by the focal
        length and NA of the objective. `y_bfp` is the corresponding grid of y locations, and
        `r_bfp` is the radial distance from the center of the back focal plane. r_max is the largest
        distance that falls inside the NA, but r_bfp will contain larger numbers as the back focal
        plane is sampled with a square grid. The function must return a tuple (Ex, Ey), which are
        the electric fields in the x- and y- direction, respectively, at the sample locations in the
        back focal plane. The fields may be complex, so a phase difference between x and y is
        possible. If only one polarization is used, the other return value must be None, e.g., y
        polarization would return (None, Ey). The fields are post-processed such that any part that
        falls outside of the NA is set to zero. This region is indicated by the variable `aperture`,
        which has the same shape as `x_bfp`, `y_bfp` and `r_bfp`, and for every location indicates
        whether it is inside the NA of the objective with `True` or outside the NA with `False`. The
        integer `bfp_sampling_n` is the number of samples of the back focal plane from the center to
        the edge of the NA, and is given for convenience or potential caching. This will be the
        number as passed below to `fields_focus()`
    objective : Objective
        instance of the Objective class
    bead : Bead
        instance of the Bead class
    bfp_sampling_n : int, optional
        Number of discrete steps with which the back focal plane is sampled, from the center to the
        edge. The total number of plane waves scales with the square of bfp_sampling_n, by default
        31
    num_orders : int, optional
        Number of orders that should be included in the calculation the Mie solution. If it is None
        (default), the code will use the Bead.number_of_orders() method to calculate a sufficient
        number.
    integration_order : int, optional
        The order of the integration. If no order is given, the code will determine an order based
        on the number of orders in the Mie solution. If the integration order is provided, that
        order is used for Gauss-Legendre integration. For Lebedev-Laikov integration, that order or
        the nearest higher order is used when the provided order does not match one of the available
        orders. The automatic determination of the integration order sets `integration_order` to
        `num_orders + 1` for Gauss-Legendre integration, and to `2 * num_orders` for Lebedev-Laikov
        and Clenshaw-Curtis integration. This typically leads to sufficiently accurate forces, but a
        convergence check is recommended.
    method : string, optional
        Integration method. Choices are "lebedev-laikov" (default), "gauss-legendre" and
        "clenshaw-curtis". With automatic determination of the integration order (`integration_order
        = None`), the Lebedev-Laikov scheme typically has less points to evaluate and therefore is
        faster for similar precision. However, the order is limited to 131. The Gauss-Legendre
        integration scheme is the next most efficient, but may have issues at a very high
        integration order. In that case, the Clenshaw-Curtis method may be used.

    Returns
    -------
    callable
        Returns a callable with the signature `f(bead_center: Tuple[float, float, float],
        num_threads: int) -> Tuple[float, float, float]`. The parameter `bead_center` is the bead
        location in space, for the x-, y-, and z-axis respectively, and is specified in meters. The
        parameter `num_threads` is the number of threads to use for the calculation. It is limited
        by `numba.config.NUMBA_NUM_THREADS`.

        The return value of a function call is the force on the bead at the specifed location, in
        Newton, in the x-, y- and z-direction.

    Raises
    ------
    ValueError
        Raised if the medium surrounding the bead does not match the immersion medium of the
        objective. Raised when an invalid integration method was specified.
    """
    if bead.n_medium != objective.n_medium:
        raise ValueError("The immersion medium of the bead and the objective have to be the same")

    n_orders = bead.number_of_orders if num_orders is None else max(int(num_orders), 1)

    if integration_order is not None:
        # Use user's integration order
        integration_order = np.max((2, int(integration_order)))
        if method == "lebedev-laikov":
            # Find nearest integration order that is equal or greater than the user's
            integration_order = get_nearest_order(integration_order)
    else:
        integration_order = determine_integration_order(method, n_orders)
    x, y, z, w = get_integration_locations(integration_order, method)
    xb, yb, zb = [c * bead.bead_diameter * 0.51 for c in (x, y, z)]

    local_coordinates = LocalBeadCoordinates(
        xb, yb, zb, bead.bead_diameter, (0.0, 0.0, 0.0), grid=False
    )
    external_fields_func = focus_field_factory(
        objective,
        bead,
        n_orders,
        bfp_sampling_n,
        f_input_field,
        local_coordinates,
        False,
    )
    _eps = EPS0 * bead.n_medium**2
    _mu = MU0

    # Normal vectors with weight factor incorporated
    nw = w[:, np.newaxis] * np.concatenate([np.atleast_2d(ax) for ax in (x, y, z)], axis=0).T

    def force_on_bead(bead_center: Tuple[float, float, float], num_threads: Optional[int] = None):
        bead_center = np.atleast_2d(bead_center)
        Ex, Ey, Ez, Hx, Hy, Hz = external_fields_func(bead_center, True, True, True, num_threads)

        Te11 = np.atleast_2d(_eps * 0.5 * (np.abs(Ex) ** 2 - np.abs(Ey) ** 2 - np.abs(Ez) ** 2))
        Te22 = np.atleast_2d(_eps * 0.5 * (np.abs(Ey) ** 2 - np.abs(Ex) ** 2 - np.abs(Ez) ** 2))
        Te33 = np.atleast_2d(_eps * 0.5 * (np.abs(Ez) ** 2 - np.abs(Ey) ** 2 - np.abs(Ex) ** 2))

        Te12 = np.atleast_2d(_eps * np.real(Ex * np.conj(Ey)))
        Te13 = np.atleast_2d(_eps * np.real(Ex * np.conj(Ez)))
        Te23 = np.atleast_2d(_eps * np.real(Ey * np.conj(Ez)))

        Th11 = np.atleast_2d(_mu * 0.5 * (np.abs(Hx) ** 2 - np.abs(Hy) ** 2 - np.abs(Hz) ** 2))
        Th22 = np.atleast_2d(_mu * 0.5 * (np.abs(Hy) ** 2 - np.abs(Hx) ** 2 - np.abs(Hz) ** 2))
        Th33 = np.atleast_2d(_mu * 0.5 * (np.abs(Hz) ** 2 - np.abs(Hy) ** 2 - np.abs(Hx) ** 2))

        Th12 = np.atleast_2d(_mu * np.real(Hx * np.conj(Hy)))
        Th13 = np.atleast_2d(_mu * np.real(Hx * np.conj(Hz)))
        Th23 = np.atleast_2d(_mu * np.real(Hy * np.conj(Hz)))

        # F = 1/2 Int Re[T dot n] da ~ 4 pi r^2 Sum w * T dot n, with r the diameter of the
        # imaginary sphere surrounding the bead on which the force (density) is calculated

        T = np.empty((len(bead_center), x.size, 3, 3))
        T[..., 0, 0] = Te11 + Th11
        T[..., 0, 1] = Te12 + Th12
        T[..., 0, 2] = Te13 + Th13
        T[..., 1, 0] = Te12 + Th12
        T[..., 1, 1] = Te22 + Th22
        T[..., 1, 2] = Te23 + Th23
        T[..., 2, 0] = Te13 + Th13
        T[..., 2, 1] = Te23 + Th23
        T[..., 2, 2] = Te33 + Th33

        # T is [num_bead_positions, num_coords_around_bead, 3, 3] large, nw is [num_coords, 3] large
        # F is the 3D force on the bead at each position `num_bead_positions`, except for the factor
        # 1/2 * 4 pi r^2
        F = np.matmul(T, nw[np.newaxis, ..., np.newaxis])[..., 0].sum(axis=1)

        # Note: the factor 1/2 is incorporated as 2 pi instead of 4 pi
        return np.squeeze(F) * (bead.bead_diameter * 0.51) ** 2 * 2 * np.pi

    return force_on_bead


def forces_focus(
    f_input_field,
    objective,
    bead,
    bead_center=(0, 0, 0),
    bfp_sampling_n=31,
    num_orders=None,
    integration_order=None,
    method="lebedev-laikov",
):
    """
    Calculate the forces on a bead in the focus of an arbitrary input
    beam, going through an objective with a certain NA and focal length.
    Implemented with the angular spectrum of plane waves and Mie theory.

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes
    electric fields in the x, y, and z directions. The input can be a
    combination of x- and y-polarized light of complex amplitudes.

    Parameters
    ----------
    f_input_field : function with signature f(X_BFP, Y_BFP, R, Rmax,
        cosTheta, cosPhi, sinPhi), where X_BFP is a grid of x locations in
        the back focal plane, determined by the focal length and NA of the
        objective. Y_BFP is the corresponding grid of y locations, and R is
        the radial distance from the center of the back focal plane. Rmax is
        the largest distance that falls inside the NA, but R will contain
        larger numbers as the back focal plane is sampled with a square
        grid. Theta is defined as the angle with the negative optical axis
        (-z), and cosTheta is the cosine of this angle. Phi is defined as
        the angle between the x and y axis, and cosPhi and sinPhi are its
        cosine and sine, respectively. The function must return a tuple
        (E_BFP_x, E_BFP_y), which are the electric fields in the x- and y-
        direction, respectively, at the sample locations in the back focal
        plane. The fields may be complex, so a phase difference between x
        and y is possible. If only one polarization is used, the other
        return value must be None, e.g., y polarization would return (None,
        E_BFP_y). The fields are post-processed such that any part that
        falls outside of the NA is set to zero.
    objective : instance of the Objective class
    bead : instance of the Bead class
    bead_center : tuple of three floating point numbers determining the
        x, y and z position of the bead center in 3D space, in meters
    bfp_sampling_n : (Default value = 31) Number of discrete steps with
        which the back focal plane is sampled, from the center to the edge.
        The total number of plane waves scales with the square of
        bfp_sampling_n num_orders: number of order that should be included
        in the calculation the Mie solution. If it is None (default), the
        code will use the number_of_orders() method to calculate a
        sufficient number.

    Returns
    -------
    F : an array with the force on the bead in the x direction F[0], in the
        y direction F[1] and in the z direction F[2]. The force is in [N].
    """
    force_fun = force_factory(
        f_input_field=f_input_field,
        objective=objective,
        bead=bead,
        bfp_sampling_n=bfp_sampling_n,
        num_orders=num_orders,
        integration_order=integration_order,
        method=method,
    )
    return force_fun(bead_center)


def absorbed_power_focus(
    f_input_field,
    objective,
    bead: Bead,
    bead_center=(0, 0, 0),
    bfp_sampling_n=31,
    num_orders=None,
    integration_order=None,
    method="lebedev-laikov",
):
    """
    Calculate the dissipated power in a bead in the focus of an arbitrary
    input beam, going through an objective with a certain NA and focal
    length. Implemented with the angular spectrum of plane waves and Mie
    theory.

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes
    electric fields in the x, y, and z directions. The input can be a
    combination of x- and y-polarized light of complex amplitudes.

    Parameters
    ----------
    f_input_field : function with signature f(X_BFP, Y_BFP, R, Rmax,
        cosTheta, cosPhi, sinPhi), where X_BFP is a grid of x locations in
        the back focal plane, determined by the focal length and NA of the
        objective. Y_BFP is the corresponding grid of y locations, and R is
        the radial distance from the center of the back focal plane. Rmax
        is the largest distance that falls inside the NA, but R will
        contain larger numbers as the back focal plane is sampled with a
        square grid. Theta is defined as the angle with the negative
        optical axis (-z), and cosTheta is the cosine of this angle. Phi is
        defined as the angle between the x and y axis, and cosPhi and
        sinPhi are its cosine and sine, respectively. The function must
        return a tuple (E_BFP_x, E_BFP_y), which are the electric fields in
        the x- and y- direction, respectively, at the sample locations in
        the back focal plane. The fields may be complex, so a phase
        difference between x and y is possible. If only one polarization is
        used, the other return value must be None, e.g., y polarization
        would return (None, E_BFP_y). The fields are post-processed such
        that any part that falls outside of the NA is set to zero.
    n_bfp : refractive index at the back focal plane of the objective
        focused focal_length: focal length of the objective, in meters
    focal_length : focal length of the objective, in meters
    NA : Numerical Aperture n_medium * sin(theta_max) of the objective
    bead_center : tuple
    of three floating point numbers determining the
        x, y and z position of the bead center in 3D space, in meters
    bfp_sampling_n : (Default value = 31) Number of discrete steps with
        which the back focal plane is sampled, from the center to the edge.
        The total number of plane waves scales with the square of
        bfp_sampling_n num_orders: number of order that should be included
        in the calculation the Mie solution. If it is None (default), the
        code will use the number_of_orders() method to calculate a
        sufficient number.

    Returns
    -------
    Pabs : the absorbed power in Watts.
    """

    n_orders = bead.number_of_orders if num_orders is None else max(int(num_orders), 1)

    if integration_order is not None:
        # Use user's integration order
        integration_order = np.max((2, int(integration_order)))
        if method == "lebedev-laikov":
            # Find nearest integration order that is equal or greater than the user's
            integration_order = get_nearest_order(integration_order)
    else:
        integration_order = determine_integration_order(method, n_orders)
    x, y, z, w = get_integration_locations(integration_order, method)
    xb, yb, zb = [
        ax * bead.bead_diameter * 0.51 + bead_center[idx] for idx, ax in enumerate((x, y, z))
    ]

    Ex, Ey, Ez, Hx, Hy, Hz = fields_focus(
        f_input_field,
        objective,
        bead,
        bead_center,
        xb,
        yb,
        zb,
        bfp_sampling_n,
        num_orders,
        return_grid=False,
        total_field=True,
        magnetic_field=True,
        verbose=False,
        grid=False,
    )

    # Note: factor 1/2 incorporated later as 2 pi instead of 4 pi
    Px = (np.conj(Hz) * Ey - np.conj(Hy) * Ez).real
    Py = (np.conj(Hx) * Ez - np.conj(Hz) * Ex).real
    Pz = (np.conj(Hy) * Ex - np.conj(Hx) * Ey).real

    # Integral of - dot(P, n) over spherical surface
    Pabs = -np.sum((Px * x + Py * y + Pz * z) * w)

    return Pabs * (bead.bead_diameter * 0.51) ** 2 * 2 * np.pi


def scattered_power_focus(
    f_input_field,
    objective: Objective,
    bead: Bead,
    bead_center=(0, 0, 0),
    bfp_sampling_n=31,
    num_orders=None,
    integration_order=None,
    method="lebedev-laikov",
):
    """
    Calculate the scattered power by a bead in the focus of an arbitrary
    input beam, going through an objective with a certain NA and focal length.
    Implemented with the angular spectrum of plane waves and Mie theory.

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes
    electric fields in the x, y, and z directions. The input can be a
    combination of x- and y-polarized light of complex amplitudes.

    Parameters
    ----------
    f_input_field : function with signature f(X_BFP, Y_BFP, R, Rmax,
        cosTheta, cosPhi, sinPhi), where X_BFP is a grid of x locations in the
        back focal plane, determined by the focal length and NA of the
        objective. Y_BFP is the corresponding grid of y locations, and R is the
        radial distance from the center of the back focal plane. Rmax is the
        largest distance that falls inside the NA, but R will contain larger
        numbers as the back focal plane is sampled with a square grid. Theta is
        defined as the angle with the negative optical axis (-z), and cosTheta
        is the cosine of this angle. Phi is defined as the angle between the x
        and y axis, and cosPhi and sinPhi are its cosine and sine,
        respectively. The function must return a tuple (E_BFP_x, E_BFP_y),
        which are the electric fields in the x- and y- direction, respectively,
        at the sample locations in the back focal plane. The fields may be
        complex, so a phase difference between x and y is possible. If only one
        polarization is used, the other return value must be None, e.g., y
        polarization would return (None, E_BFP_y). The fields are
        post-processed such that any part that falls outside of the NA is set
        to zero.
    n_bfp : refractive index at the back focal plane of the objective
        focused focal_length: focal length of the objective, in meters
    focal_length : focal length of the objective, in meters NA : Numerical
    Aperture n_medium * sin(theta_max) of the objective bead_center : tuple of
    three floating point numbers determining the
        x, y and z position of the bead center in 3D space, in meters
    bfp_sampling_n : (Default value = 31) Number of discrete steps with
        which the back focal plane is sampled, from the center to the edge. The
        total number of plane waves scales with the square of bfp_sampling_n
        num_orders: number of order that should be included in the calculation
        the Mie solution. If it is None (default), the code will use the
        number_of_orders() method to calculate a sufficient number.

    Returns
    -------
    Psca : the scattered power in Watts.
    """

    n_orders = bead.number_of_orders if num_orders is None else max(int(num_orders), 1)

    if integration_order is not None:
        # Use user's integration order
        integration_order = np.max((2, int(integration_order)))
        if method == "lebedev-laikov":
            # Find nearest integration order that is equal or greater than the user's
            integration_order = get_nearest_order(integration_order)
    else:
        integration_order = determine_integration_order(method, n_orders)

    x, y, z, w = get_integration_locations(integration_order, method)

    xb, yb, zb = [
        ax * bead.bead_diameter * 0.51 + bead_center[idx] for idx, ax in enumerate((x, y, z))
    ]

    Ex, Ey, Ez, Hx, Hy, Hz = fields_focus(
        f_input_field,
        objective,
        bead,
        bead_center,
        xb,
        yb,
        zb,
        bfp_sampling_n,
        num_orders,
        return_grid=False,
        total_field=False,
        magnetic_field=True,
        verbose=False,
        grid=False,
    )

    # Note: factor 1/2 incorporated later as 2 pi instead of 4 pi
    Px = (np.conj(Hz) * Ey - np.conj(Hy) * Ez).real
    Py = (np.conj(Hx) * Ez - np.conj(Hz) * Ex).real
    Pz = (np.conj(Hy) * Ex - np.conj(Hx) * Ey).real

    # integral of dot(P, n) over spherical surface:
    Psca = np.sum((Px * x + Py * y + Pz * z) * w)

    return Psca * (bead.bead_diameter * 0.51) ** 2 * 2 * np.pi

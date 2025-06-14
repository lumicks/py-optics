import logging
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
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
    filling_factor: float,
    objective: Objective,
    bead: Bead,
    bead_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    x: float | ArrayLike = 0.0,
    y: float | ArrayLike = 0.0,
    z: float | ArrayLike = 0.0,
    *,
    return_grid: bool = False,
    total_field: bool = True,
    magnetic_field: bool = False,
    verbose: bool = False,
    grid: bool = True,
    num_spherical_modes: int | None = None,
    integration_order_bfp: int | None = None,
    integration_method_bfp: str = "peirce",
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
    x : float | ArrayLike
        array of x locations for evaluation, in meters
    y : float | ArrayLike
        array of y locations for evaluation, in meters
    z : float | ArrayLike
        array of z locations for evaluation, in meters
    bead_center : tuple[float, float, float]
        A tuple of three floating point numbers determining the x, y and z position of the bead
        center in 3D space, in meters
    return_grid: bool, optional
        Return the sampling grid (the result of np.meshgrid) in the matrices X, Y and Z. Default
        is False
    total_field : bool, optional
        If True, return the total field (sum) of incident and scattered electromagnetic field
        (default). If False, then only return the scattered field outside the bead. Inside the bead,
        the full field is always returned.
    magnetic_field : bool, optional
        If True, return the magnetic fields and the electric fields. If false (default), do not
        return the magnetic fields, but only the electric fields.
    verbose : bool, optional
        If True, print statements on the progress of the calculation. Default is False
    grid: bool, optional
        If True (default), interpret the vectors or scalars x, y and z as the input for the
        numpy.meshgrid function, and calculate the fields at the locations that are the result of
        the numpy.meshgrid output. If False, interpret the x, y and z vectors as the exact locations
        where the field needs to be evaluated. In that case, all vectors need to be of the same
        length.
    num_spherical_modes : int, optional
        Number of order that should be included in the calculation the Mie solution. If it is `None`
        (default), the code will use the :py:method:`Bead.number_of_modes()` method to calculate a
        sufficient number.
    integration_order_bfp : int | None
        Order of the method to integrate over the back focal plane of the objective. Typically, the
        higher the order, the more accurate the result, or the region of convergence expands. If
        `None`, the function tries to figure out a reasonable default value for the integration
        methods "peirce" (default, see below) and "equidistant". For other integration methods the
        order has to be set by hand.
    integration_method_bfp : str, optional
        The method of integrating over the back focal plane. The default is "peirce", which is a
        Gaussian-Legendre-like integration method. Other options are "lether", "equidistant" and
        "takaki", see notes below.

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
    w0 = filling_factor * objective.r_bfp_max  # [m]
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
        num_spherical_modes=num_spherical_modes,
        return_grid=return_grid,
        total_field=total_field,
        magnetic_field=magnetic_field,
        verbose=verbose,
        grid=grid,
        integration_order_bfp=integration_order_bfp,
        integration_method_bfp=integration_method_bfp,
    )


def fields_focus(
    f_input_field: Callable,
    objective: Objective,
    bead: Bead,
    bead_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    x: float | ArrayLike = 0.0,
    y: float | ArrayLike = 0.0,
    z: float | ArrayLike = 0.0,
    *,
    return_grid: bool = False,
    total_field: bool = True,
    magnetic_field: bool = False,
    verbose: bool = False,
    grid: bool = True,
    num_spherical_modes: int | None = None,
    integration_order_bfp: int | None = None,
    integration_method_bfp: str = "peirce",
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
    f_input_field : Callable
        function with signature `f(coordinates: BackFocalPlaneCoordinates, objective: Objective)`,
        see :py:class:`pyoptics.objective.BackFocalPlaneCoordinates` and
        :py:class:`pyoptics.objective.Objective`. The function must return a tuple (Ex, Ey), which
        are the electric fields in the x- and y- direction, respectively, at the sample locations in
        the back focal plane. The fields may be complex, so a phase difference between x and y is
        possible. If only one polarization is used, the other return value must be None, e.g., y
        polarization would return (None, Ey). The fields are post-processed such that any part that
        falls outside of the NA is set to zero.
    objective : Objective
        Instance of the Objective class
    bead : Bead
        Instance of the Bead class
    x : float | ArrayLike
        array of x locations for evaluation, in meters
    y : float | ArrayLike
        array of y locations for evaluation, in meters
    z : float | ArrayLike
        array of z locations for evaluation, in meters
    bead_center : tuple[float, float, float]
        A tuple of three floating point numbers determining the x, y and z position of the bead
        center in 3D space, in meters
    return_grid: bool, optional
        Return the sampling grid (the result of np.meshgrid) in the matrices X, Y and Z. Default
        is False
    total_field : bool, optional
        If True, return the total field (sum) of incident and scattered electromagnetic field
        (default). If False, then only return the scattered field outside the bead. Inside the bead,
        the full field is always returned.
    magnetic_field : bool, optional
        If True, return the magnetic fields and the electric fields. If false (default), do not
        return the magnetic fields, but only the electric fields.
    verbose : bool, optional
        If True, print statements on the progress of the calculation. Default is False
    grid: bool, optional
        If True (default), interpret the vectors or scalars x, y and z as the input for the
        numpy.meshgrid function, and calculate the fields at the locations that are the result of
        the numpy.meshgrid output. If False, interpret the x, y and z vectors as the exact locations
        where the field needs to be evaluated. In that case, all vectors need to be of the same
        length.
    num_spherical_modes : int
        Number of order that should be included in the calculation the Mie solution. If it is `None`
        (default), the code will use the :py:method:`Bead.number_of_modes()` method to calculate a
        sufficient number.
    integration_order_bfp : int | None
        Order of the method to integrate over the back focal plane of the objective. Typically, the
        higher the order, the more accurate the result, or the region of convergence of the area
        around the focus expands. If `None`, the function tries to figure out a reasonable default
        value for the integration methods "peirce" (default, see below) and "equidistant". For other
        integration methods the order has to be set by hand.
    integration_method_bfp : str
        The method of integrating over the back focal plane. The default is "peirce", which is a
        Gaussian-Legendre-like integration method. Other options are "lether", "equidistant" and
        "takaki", see notes below.

    Raises
    ------
    ValueError:
        Raised when the immersion medium of the bead does not match the medium of the objective.

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

    num_spherical_modes = (
        bead.number_of_modes if num_spherical_modes is None else max(int(num_spherical_modes), 1)
    )
    logging.info(f"Using {num_spherical_modes} spherical modes")
    local_coordinates = LocalBeadCoordinates(x, y, z, bead.bead_diameter, bead_center, grid=grid)

    if integration_order_bfp is None:
        integration_order_bfp = objective.minimal_integration_order(
            [x, y, z], lambda_vac=bead.lambda_vac, method=integration_method_bfp
        )

    logging.info("Calculating auxiliary data for external fields")
    field_fun = focus_field_factory(
        objective=objective,
        bead=bead,
        num_spherical_modes=num_spherical_modes,
        integration_order_bfp=integration_order_bfp,
        integration_method_bfp=integration_method_bfp,
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
        num_spherical_modes=num_spherical_modes,
        integration_order_bfp=integration_order_bfp,
        integration_method_bfp=integration_method_bfp,
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
        X, Y, Z = (np.squeeze(axis) for axis in np.meshgrid(x, y, z, indexing="ij"))
        ret += (X, Y, Z)

    return ret


def fields_plane_wave(
    bead: Bead,
    x: float | ArrayLike,
    y: float | ArrayLike,
    z: float | ArrayLike,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: tuple[float | complex, float | complex] = (1.0, 0.0),
    num_spherical_modes: int | None = None,
    return_grid: bool = False,
    total_field: bool = True,
    magnetic_field: bool = False,
    verbose: bool = False,
    grid: bool = True,
):
    """
    Calculate the electromagnetic field of a bead, subject to excitation by a plane wave. The plane
    wave can be at an angle theta and phi, and have a polarization state that is the combination of
    the (complex) amplitudes of a theta-polarization state and phi-polarized state. If theta == 0
    and phi == 0, the theta polarization points along +x axis and the wave travels into the +z
    direction.

    Parameters
    ----------
    bead: Bead
        Instance of the Bead class
    x : float | ArrayLike
        Array of x locations for evaluation, in meters
    y : float | ArrayLike
        Array of y locations for evaluation, in meters
    z : float | ArrayLike
        Array of z locations for evaluation, in meters
    theta : float
        Angle with the negative optical axis (-z), radians
    phi : float
        Angle with the positive x axis, radians
    num_spherical_modes : int
        Number of order that should be included in the calculation the Mie solution. If it is None
        (default), the code will use the number_of_modes() method to calculate a sufficient number.
    return_grid : bool
        Return the sampling grid in the matrices X, Y and Z, default = False
    total_field : bool
        If True, return the total field of incident and scattered electromagnetic field (default).
        If False, then only return the scattered field outside the bead. Inside the bead, the full
        field is always returned.
    magnetic_field : bool
        If True, return the magnetic fields as well. If false (default), do not return the magnetic
        fields. verbose : If True, print statements on the progress of the calculation. Default is
        False
    grid : bool
        If True (default), interpret the vectors or scalars x, y and z as the input for the
        numpy.meshgrid function, and calculate the fields at the locations that are the result of
        the numpy.meshgrid output. If False, interpret the x, y and z vectors as the exact locations
        where the field needs to be evaluated. In that case, all vectors need to be of the same
        length.

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

    n_orders = (
        bead.number_of_modes if num_spherical_modes is None else max(int(num_spherical_modes), 1)
    )
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
    f_input_field: Callable,
    objective: Objective,
    bead: Bead,
    integration_order_bfp: int,
    *,
    integration_method_bfp: str = "peirce",
    num_spherical_modes: int | None = None,
    spherical_integration_order: int | None = None,
    spherical_integration_method: str = "lebedev-laikov",
) -> Callable:
    """Create and return a function suitable to calculate the force on a bead. Items that can be
    precalculated are stored for rapid subsequent calculations of the force on the bead for
    different positions.

    Parameters
    ----------
    f_input_field : callable
        function with signature `f(coordinates: BackFocalPlaneCoordinates, objective: Objective)`,
        see :py:class:`pyoptics.objective.BackFocalPlaneCoordinates` and
        :py:class:`pyoptics.objective.Objective`. The function must return a tuple (Ex, Ey), which
        are the electric fields in the x- and y- direction, respectively, at the sample locations in
        the back focal plane. The fields may be complex, so a phase difference between x and y is
        possible. If only one polarization is used, the other return value must be None, e.g., y
        polarization would return (None, Ey). The fields are post-processed such that any part that
        falls outside of the NA is set to zero.
    objective : Objective
        instance of the Objective class
    bead : Bead
        instance of the Bead class
    integration_order_bfp : int
        Order of the method to integrate over the back focal plane of the objective. Typically, the
        higher the order, the more accurate the result, or the region of convergence of the area
        around the focus expands. The order has to be provided, but if the region of interest is
        known in advance, the :py:method:`Objective.minimal_integration_order` method can help to
        determine a reasonable starting value, though it only supports the integration methods
        "peirce" and "equidistant" (see below).
    integration_method_bfp : str
        The method of integrating over the back focal plane. The default is "peirce", which is a
        Gaussian-Legendre-like integration method. Other options are "lether", "equidistant" and
        "takaki", see notes below.
    num_spherical_modes : int
        Number of orders that should be included in the calculation the Mie solution. If it is None
        (default), the code will use the Bead.number_of_modes() method to calculate a sufficient
        number.
    spherical_integration_order : int, optional
        The order of the integration. If no order is given, the code will determine an order based
        on the number of orders in the Mie solution. If the integration order is provided, that
        order is used for Gauss-Legendre integration. For Lebedev-Laikov integration, that order or
        the nearest higher order is used when the provided order does not match one of the available
        orders. The automatic determination of the integration order sets `integration_order` to
        `num_spherical_modes + 1` for Gauss-Legendre integration, and to `2 * num_spherical_modes`
        for Lebedev-Laikov and Clenshaw-Curtis integration. This typically leads to sufficiently
        accurate forces, but a convergence check is recommended.
    spherical_integration_method : string, optional
        Integration method. Choices are "lebedev-laikov" (default), "gauss-legendre" and
        "clenshaw-curtis". With automatic determination of the integration order (`integration_order
        = None`), the Lebedev-Laikov scheme typically has less points to evaluate and therefore is
        faster for similar precision. However, the order is limited to 131. The Gauss-Legendre
        integration scheme is the next most efficient, but may have issues at a very high
        integration order. In that case, the Clenshaw-Curtis method may be used.

    Returns
    -------
    Callable
        Returns a callable with the signature `f(bead_center: tuple[float, float,
        float]|list[tuple[float, float, float]], num_threads: int) -> tuple[float, float,
        float]|list[tuple[float, float, float]]`. The parameter `bead_center` is the bead location
        in space, for the x-, y-, and z-axis respectively, and is specified in meters. The parameter
        `num_threads` is the number of threads to use for the calculation. The upper limit is set by
        `numba.config.NUMBA_NUM_THREADS`.

        The return value of a function call is (a list of) the force on the bead at the specifed
        location, in Newton, in the x-, y- and z-direction.

    Raises
    ------
    ValueError
        Raised if the medium surrounding the bead does not match the immersion medium of the
        objective. Raised when an invalid integration method was specified.
    """
    if bead.n_medium != objective.n_medium:
        raise ValueError("The immersion medium of the bead and the objective have to be the same")

    n_orders = (
        bead.number_of_modes
        if num_spherical_modes is None
        else max(math.floor(num_spherical_modes), 1)
    )

    if spherical_integration_order is not None:
        # Use user's integration order
        spherical_integration_order = np.max((2, int(spherical_integration_order)))
        # Find nearest integration order that is equal or greater than the user's
        spherical_integration_order = get_nearest_order(
            spherical_integration_order, spherical_integration_method
        )
    else:
        spherical_integration_order = _determine_integration_order(
            spherical_integration_method, n_orders
        )
    x, y, z, w = get_integration_locations(
        spherical_integration_order, spherical_integration_method
    )
    xb, yb, zb = [c * bead.bead_diameter * 0.51 for c in (x, y, z)]

    local_coordinates = LocalBeadCoordinates(
        xb, yb, zb, bead.bead_diameter, (0.0, 0.0, 0.0), grid=False
    )
    external_fields_func = focus_field_factory(
        objective,
        bead,
        f_input_field=f_input_field,
        local_coordinates=local_coordinates,
        num_spherical_modes=n_orders,
        integration_order_bfp=integration_order_bfp,
        integration_method_bfp=integration_method_bfp,
        internal=False,
    )
    _eps = EPS0 * bead.n_medium**2
    _mu = MU0

    # Normal vectors with weight factor incorporated
    nw = w[:, np.newaxis] * np.concatenate([np.atleast_2d(ax) for ax in (x, y, z)], axis=0).T

    def force_on_bead(bead_center: tuple[float, float, float], num_threads: int | None = None):
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
    f_input_field: Callable,
    objective: Objective,
    bead: Bead,
    bead_center: tuple[float, float, float] | list[tuple[float, float, float]] = (0.0, 0.0, 0.0),
    *,
    num_spherical_modes: int | None = None,
    integration_order_bfp: int | None = None,
    integration_method_bfp: str = "peirce",
    spherical_integration_order: int | None = None,
    spherical_integration_method: str = "lebedev-laikov",
):
    """
    Calculate the forces on a bead in the focus of an arbitrary input beam, going through an
    objective with a certain NA and focal length. Implemented with the angular spectrum of plane
    waves and Mie theory.

    This function correctly incorporates the polarized nature of light in a focus. In other words,
    the polarization state at the focus includes electric fields in the x, y, and z directions. The
    input can be a combination of x- and y-polarized light of complex amplitudes.

    Parameters
    ----------
    f_input_field : Callable
        function with signature `f(coordinates: BackFocalPlaneCoordinates, objective: Objective)`,
        see :py:class:`pyoptics.objective.BackFocalPlaneCoordinates` and
        :py:class:`pyoptics.objective.Objective`. The function must return a tuple (Ex, Ey), which
        are the electric fields in the x- and y- direction, respectively, at the sample locations in
        the back focal plane. The fields may be complex, so a phase difference between x and y is
        possible. If only one polarization is used, the other return value must be None, e.g., y
        polarization would return (None, Ey). The fields are post-processed such that any part that
        falls outside of the NA is set to zero.
    objective : Objective
        Instance of the Objective class
    bead : Bead
        Instance of the Bead class
    bead_center : list[tuple[float, float, float]] | tuple[float, float, float]
        (List of) three floating point numbers determining the x, y and z position of the bead
        center in 3D space, in meters.
    num_spherical_modes : int
        Number of order that should be included in the calculation the Mie solution. If it is `None`
        (default), the code will use the :py:method:`Bead.number_of_modes()` method to calculate a
        sufficient number.
    integration_order_bfp : int | None
        Order of the method to integrate over the back focal plane of the objective. Typically, the
        higher the order, the more accurate the result, or the region of convergence of the area
        around the focus expands. If `None`, the function tries to figure out a reasonable default
        value for the integration methods "peirce" (default, see below) and "equidistant". For other
        integration methods the order has to be set by hand.
    integration_method_bfp : str
        The method of integrating over the back focal plane. The default is "peirce", which is a
        Gaussian-Legendre-like integration method. Other options are "lether", "equidistant" and
        "takaki", see notes below.
    spherical_integration_order : int, optional
        The order of the integration. If no order is given, the code will determine an order based
        on the number of orders in the Mie solution. If the integration order is provided, that
        order is used for Gauss-Legendre integration. For Lebedev-Laikov integration, that order or
        the nearest higher order is used when the provided order does not match one of the available
        orders. The automatic determination of the integration order sets `integration_order` to
        `num_spherical_modes + 1` for Gauss-Legendre integration, and to `2 * num_spherical_modes`
        for Lebedev-Laikov and Clenshaw-Curtis integration. This typically leads to sufficiently
        accurate forces, but a convergence check is recommended.
    spherical_integration_method : string, optional
        Integration method. Choices are "lebedev-laikov" (default), "gauss-legendre" and
        "clenshaw-curtis". With automatic determination of the integration order (`integration_order
        = None`), the Lebedev-Laikov scheme typically has less points to evaluate and therefore is
        faster for similar precision. However, the order is limited to 131. The Gauss-Legendre
        integration scheme is the next most efficient, but may have issues at a very high
        integration order. In that case, the Clenshaw-Curtis method may be used.


    Returns
    -------
    F : an array with the force on the bead in the x direction F[0], in the
        y direction F[1] and in the z direction F[2]. If a list of bead positions is passed, then
        the forces are like F[ax, :] with ax = 0..2. The force is in [N].
    """
    force_fun = force_factory(
        f_input_field=f_input_field,
        objective=objective,
        bead=bead,
        integration_order_bfp=integration_order_bfp,
        integration_method_bfp=integration_method_bfp,
        num_spherical_modes=num_spherical_modes,
        spherical_integration_order=spherical_integration_order,
        spherical_integration_method=spherical_integration_method,
    )
    return force_fun(bead_center)


def _scattered_absorbed_power_focus(
    f_input_field,
    objective,
    bead: Bead,
    bead_center,
    integration_order_bfp,
    integration_method_bfp,
    num_spherical_modes,
    spherical_integration_order,
    spherical_integration_method,
    absorbed: bool,
):
    n_orders = (
        bead.number_of_modes if num_spherical_modes is None else max(int(num_spherical_modes), 1)
    )

    if spherical_integration_order is not None:
        # Use user's integration order
        spherical_integration_order = np.max((2, int(spherical_integration_order)))
        # Find nearest integration order that is equal or greater than the user's
        spherical_integration_order = get_nearest_order(
            spherical_integration_order, spherical_integration_method
        )
    else:
        integration_order = _determine_integration_order(spherical_integration_method, n_orders)
    x, y, z, w = get_integration_locations(integration_order, spherical_integration_method)
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
        integration_order_bfp=integration_order_bfp,
        integration_method_bfp=integration_method_bfp,
        num_spherical_modes=num_spherical_modes,
        return_grid=False,
        total_field=absorbed,
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


def absorbed_power_focus(
    f_input_field,
    objective,
    bead: Bead,
    bead_center: tuple[float, float, float] | list[tuple[float, float, float]] = (0.0, 0.0, 0.0),
    integration_order_bfp=None,
    integration_method_bfp="peirce",
    num_spherical_modes=None,
    spherical_integration_order=None,
    spherical_integration_method="lebedev-laikov",
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
    f_input_field : Callable
        function with signature `f(coordinates: BackFocalPlaneCoordinates, objective: Objective)`,
        see :py:class:`pyoptics.objective.BackFocalPlaneCoordinates` and
        :py:class:`pyoptics.objective.Objective`. The function must return a tuple (Ex, Ey), which
        are the electric fields in the x- and y- direction, respectively, at the sample locations in
        the back focal plane. The fields may be complex, so a phase difference between x and y is
        possible. If only one polarization is used, the other return value must be None, e.g., y
        polarization would return (None, Ey). The fields are post-processed such that any part that
        falls outside of the NA is set to zero.
    n_bfp : float
        Refractive index at the back focal plane of the objective focused
    focal_length : float
        focal length of the objective, in meters
    NA : float
        Numerical Aperture n_medium * sin(theta_max) of the objective
    bead_center : list[tuple[float, float, float]] | tuple[float, float, float]
        (List of) three floating point numbers determining the x, y and z position of the bead
        center in 3D space, in meters.
    num_spherical_modes : int
        Number of order that should be included in the calculation the Mie solution. If it is `None`
        (default), the code will use the :py:method:`Bead.number_of_modes()` method to calculate a
        sufficient number.
    integration_order_bfp : int | None
        Order of the method to integrate over the back focal plane of the objective. Typically, the
        higher the order, the more accurate the result, or the region of convergence of the area
        around the focus expands. If `None`, the function tries to figure out a reasonable default
        value for the integration methods "peirce" (default, see below) and "equidistant". For other
        integration methods the order has to be set by hand.
    integration_method_bfp : str
        The method of integrating over the back focal plane. The default is "peirce", which is a
        Gaussian-Legendre-like integration method. Other options are "lether", "equidistant" and
        "takaki", see notes below.
    spherical_integration_order : int, optional
        The order of the integration. If no order is given, the code will determine an order based
        on the number of orders in the Mie solution. If the integration order is provided, that
        order is used for Gauss-Legendre integration. For Lebedev-Laikov integration, that order or
        the nearest higher order is used when the provided order does not match one of the available
        orders. The automatic determination of the integration order sets `integration_order` to
        `num_spherical_modes + 1` for Gauss-Legendre integration, and to `2 * num_spherical_modes`
        for Lebedev-Laikov and Clenshaw-Curtis integration. This typically leads to sufficiently
        accurate forces, but a convergence check is recommended.
    spherical_integration_method : string, optional
        Integration method. Choices are "lebedev-laikov" (default), "gauss-legendre" and
        "clenshaw-curtis". With automatic determination of the integration order (`integration_order
        = None`), the Lebedev-Laikov scheme typically has less points to evaluate and therefore is
        faster for similar precision. However, the order is limited to 131. The Gauss-Legendre
        integration scheme is the next most efficient, but may have issues at a very high
        integration order. In that case, the Clenshaw-Curtis method may be used.

    Returns
    -------
    float | list[float]
        The absorbed power in Watts, as a list if multiple positions were given.
    """

    return _scattered_absorbed_power_focus(
        f_input_field,
        objective,
        bead,
        bead_center,
        integration_order_bfp,
        integration_method_bfp,
        num_spherical_modes,
        spherical_integration_order,
        spherical_integration_method,
        True,
    )


def scattered_power_focus(
    f_input_field,
    objective: Objective,
    bead: Bead,
    bead_center=(0, 0, 0),
    integration_order_bfp=None,
    integration_method_bfp="peirce",
    num_spherical_modes=None,
    spherical_integration_order=None,
    spherical_integration_method="lebedev-laikov",
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
    f_input_field : Callable
        function with signature `f(coordinates: BackFocalPlaneCoordinates, objective: Objective)`,
        see :py:class:`pyoptics.objective.BackFocalPlaneCoordinates` and
        :py:class:`pyoptics.objective.Objective`. The function must return a tuple (Ex, Ey), which
        are the electric fields in the x- and y- direction, respectively, at the sample locations in
        the back focal plane. The fields may be complex, so a phase difference between x and y is
        possible. If only one polarization is used, the other return value must be None, e.g., y
        polarization would return (None, Ey). The fields are post-processed such that any part that
        falls outside of the NA is set to zero.
    objective : Objective
        Instance of the Objective class
    bead : Bead
        Instance of the Bead class
    bead_center : list[tuple[float, float, float]] | tuple[float, float, float]
        (List of) three floating point numbers determining the x, y and z position of the bead
        center in 3D space, in meters.
    num_spherical_modes : int
        Number of order that should be included in the calculation the Mie solution. If it is `None`
        (default), the code will use the :py:method:`Bead.number_of_modes()` method to calculate a
        sufficient number.
    integration_order_bfp : int | None
        Order of the method to integrate over the back focal plane of the objective. Typically, the
        higher the order, the more accurate the result, or the region of convergence of the area
        around the focus expands. If `None`, the function tries to figure out a reasonable default
        value for the integration methods "peirce" (default, see below) and "equidistant". For other
        integration methods the order has to be set by hand.
    integration_method_bfp : str
        The method of integrating over the back focal plane. The default is "peirce", which is a
        Gaussian-Legendre-like integration method. Other options are "lether", "equidistant" and
        "takaki", see notes below.
    spherical_integration_order : int, optional
        The order of the integration. If no order is given, the code will determine an order based
        on the number of orders in the Mie solution. If the integration order is provided, that
        order is used for Gauss-Legendre integration. For Lebedev-Laikov integration, that order or
        the nearest higher order is used when the provided order does not match one of the available
        orders. The automatic determination of the integration order sets `integration_order` to
        `num_spherical_modes + 1` for Gauss-Legendre integration, and to `2 * num_spherical_modes`
        for Lebedev-Laikov and Clenshaw-Curtis integration. This typically leads to sufficiently
        accurate forces, but a convergence check is recommended.
    spherical_integration_method : string, optional
        Integration method. Choices are "lebedev-laikov" (default), "gauss-legendre" and
        "clenshaw-curtis". With automatic determination of the integration order (`integration_order
        = None`), the Lebedev-Laikov scheme typically has less points to evaluate and therefore is
        faster for similar precision. However, the order is limited to 131. The Gauss-Legendre
        integration scheme is the next most efficient, but may have issues at a very high
        integration order. In that case, the Clenshaw-Curtis method may be used.

    Returns
    -------
    float | list[float]
        The scattered power in Watts, as a list if multiple positions were given.
    """

    return _scattered_absorbed_power_focus(
        f_input_field,
        objective,
        bead,
        bead_center,
        integration_order_bfp,
        integration_method_bfp,
        num_spherical_modes,
        spherical_integration_order,
        spherical_integration_method,
        False,
    )

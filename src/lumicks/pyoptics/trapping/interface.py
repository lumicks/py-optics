import numpy as np
from scipy.constants import (
    speed_of_light as _C,
    epsilon_0 as EPS0,
    mu_0 as MU0
)
import logging

from .bead import Bead
from ..mathutils.lebedev_laikov import (
    get_integration_locations,
    get_nearest_order
)
from .legendre_data import calculate_legendre
from .radial_data import (
    calculate_external,
    calculate_internal
)
from .local_coordinates import (
    LocalBeadCoordinates,
    ExternalBeadCoordinates,
    InternalBeadCoordinates,
)
from ..objective import (
    Objective,
    FarfieldData
)
from .implementation import calculate_fields


def fields_focus_gaussian(
    beam_power: float, filling_factor,
    objective: Objective,
    bead: Bead, bead_center=(0, 0, 0),
    x=0, y=0, z=0,
    bfp_sampling_n=31, num_orders=None,
    return_grid=False,
    total_field=True, magnetic_field=False,
    verbose=False,
    grid=True
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
    w0 = (
        filling_factor * objective.focal_length * objective.NA / bead.n_medium
    )  # [m]
    I0 = 2 * beam_power / (np.pi * w0**2)  # [W/m^2]
    E0 = (I0 * 2/(EPS0 * _C * objective.n_bfp))**0.5  # [V/m]

    def gaussian_beam(_, x_bfp, y_bfp, *args):
        Ex = np.exp(-(x_bfp**2 + y_bfp**2) / w0**2) * E0
        return (Ex, None)

    return fields_focus(
        gaussian_beam, objective, bead,
        x=x, y=y, z=z,
        bead_center=bead_center, bfp_sampling_n=bfp_sampling_n,
        num_orders=num_orders, return_grid=return_grid,
        total_field=total_field, magnetic_field=magnetic_field,
        verbose=verbose, grid=grid
    )


def fields_focus(
    f_input_field, objective: Objective,
    bead: Bead, bead_center=(0, 0, 0),
    x=0, y=0, z=0,
    bfp_sampling_n=31, num_orders=None,
    return_grid=False,
    total_field=True, magnetic_field=False,
    verbose=False,
    grid=True
):
    """
    Calculate the three-dimensional electromagnetic field of a bead in the
    focus of an arbitrary input beam, going through an objective with a certain
    NA and focal length. Implemented with the angular spectrum of plane waves
    and Mie theory.

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes
    electric fields in the x, y, and z directions. The input can be a
    combination of x- and y-polarized light of complex amplitudes.

    Parameters
    ----------
    f_input_field : function with signature `f(aperture, x_bfp, y_bfp, r_bfp,
        r_max, bfp_sampling_n)`, where `x_bfp` is a grid of x locations in the
        back focal plane, determined by the focal length and NA of the
        objective. `y_bfp` is the corresponding grid of y locations, and
        `r_bfp` is the radial distance from the center of the back focal plane.
        r_max is the largest distance that falls inside the NA, but r_bfp will
        contain larger numbers as the back focal plane is sampled with a square
        grid. The function must return a tuple (Ex, Ey), which are the electric
        fields in the x- and y- direction, respectively, at the sample
        locations in the back focal plane. The fields may be complex, so a
        phase difference between x and y is possible. If only one polarization
        is used, the other return value must be None, e.g., y polarization
        would return (None, Ey). The fields are post-processed such that any
        part that falls outside of the NA is set to zero. This region is
        indicated by the variable `aperture`, which has the same shape as
        `x_bfp`, `y_bfp` and `r_bfp`, and for every location indicates whether
        it is inside the NA of the objective with `True` or outside the NA with
        `False`. The integer `bfp_sampling_n` is the number of samples of the
        back focal plane from the center to the edge of the NA, and is given
        for convenience or potential caching. This will be the number as passed
        below to `fields_focus()`
    objective : instance of the Objective class bead : instance of the Bead
    class x : array of x locations for evaluation, in meters y : array of y
    locations for evaluation, in meters z : array of z locations for
    evaluation, in meters bead_center : tuple of three floating point numbers
    determining the x, y
        and z position of the bead center in 3D space, in meters
    bfp_sampling_n : (Default value = 31) Number of discrete steps with which
        the back focal plane is sampled, from the center to the edge. The total
        number of plane waves scales with the square of bfp_sampling_n
    num_orders: number of order that should be included in the calculation
        the Mie solution. If it is None (default), the code will use the
        number_of_orders() method to calculate a sufficient number.
    return_grid : (Default value = False) return the sampling grid in the
        matrices X, Y and Z
    total_field : If True, return the total field of incident and scattered
        electromagnetic field (default). If False, then only return the
        scattered field outside the bead. Inside the bead, the full field is
        always returned.
    magnetic_field: If True, return the magnetic fields as well. If false
        (default), do not return the magnetic fields.
    verbose: If True, print statements on the progress of the calculation.
        Default is False
    grid: If True (default), interpret the vectors or scalars x, y and z as
        the input for the numpy.meshgrid function, and calculate the fields at
        the locations that are the result of the numpy.meshgrid output. If
        False, interpret the x, y and z vectors as the exact locations where
        the field needs to be evaluated. In that case, all vectors need to be
        of the same length.

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
    if bead.n_medium != objective.n_medium:
        raise ValueError(
            'The immersion medium of the bead '
            'and the objective have to be the same'
        )

    loglevel = logging.getLogger().getEffectiveLevel()
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    # TODO: warning criteria for undersampling/aliasing
    # M = int(np.max((31, 2 * NA**2 * np.max(np.abs(z)) /
    #        (np.sqrt(self.n_medium**2 - NA**2) * self.lambda_vac))))
    # if M > bfp_sampling_n:
    #    print('bfp_sampling_n lower than recommendation for convergence')

    local_coordinates = LocalBeadCoordinates(
        x, y, z, bead.bead_diameter, bead_center, grid=grid
    )
    internal_coordinates = InternalBeadCoordinates(local_coordinates)
    external_coordinates = ExternalBeadCoordinates(local_coordinates)

    an, _ = bead.ab_coeffs(num_orders)
    n_orders = an.size

    bfp_coords, bfp_fields = objective.sample_back_focal_plane(
        f_input_field=f_input_field, bfp_sampling_n=bfp_sampling_n
    )

    farfield_data = objective.back_focal_plane_to_farfield(
        bfp_coords, bfp_fields, bead.lambda_vac
    )

    logging.info('Calculating Hankel functions and derivatives')
    external_radial_data = calculate_external(
        bead.k, external_coordinates.r, n_orders)

    logging.info(
        'Calculating Associated Legendre polynomials for external fields')
    legendre_data_ext = calculate_legendre(
        external_coordinates,
        farfield_data=farfield_data,
        n_orders=n_orders
    )

    Ex = np.zeros(local_coordinates.coordinate_shape, dtype='complex128')
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)

    if magnetic_field:
        Hx = np.zeros_like(Ex)
        Hy = np.zeros_like(Ex)
        Hz = np.zeros_like(Ex)
    else:
        Hx = 0
        Hy = 0
        Hz = 0

    logging.info('Calculating external fields')

    calculate_fields(
        Ex, Ey, Ez, Hx, Hy, Hz,
        bead=bead, bead_center=bead_center,
        local_coordinates=external_coordinates,
        farfield_data=farfield_data, legendre_data=legendre_data_ext,
        external_radial_data=external_radial_data,
        internal=False,
        total_field=total_field, magnetic_field=magnetic_field,
    )

    logging.info('Calculating Bessel functions and derivatives')
    internal_radial_data = calculate_internal(
        bead.k1, internal_coordinates.r, n_orders
    )

    logging.info(
        'Calculating Associated Legendre polynomials for internal fields')
    legendre_data_int = calculate_legendre(
        internal_coordinates,
        farfield_data=farfield_data,
        n_orders=n_orders
    )

    logging.info('Calculating internal fields')
    calculate_fields(
        Ex, Ey, Ez, Hx, Hy, Hz,
        bead=bead, bead_center=bead_center,
        local_coordinates=internal_coordinates,
        farfield_data=farfield_data, legendre_data=legendre_data_int,
        internal_radial_data=internal_radial_data,
        internal=True,
        total_field=total_field, magnetic_field=magnetic_field,
    )

    logging.getLogger().setLevel(loglevel)
    ks = bead.k * objective.NA / bead.n_medium
    dk = ks / (bfp_sampling_n - 1)
    phase = -1j * objective.focal_length * (
        np.exp(-1j * bead.k * objective.focal_length) * dk**2 / (2 * np.pi)
    )
    
    for component in (Ex, Ey, Ez):
        component *= phase

    Ex = np.squeeze(Ex)
    Ey = np.squeeze(Ey)
    Ez = np.squeeze(Ez)

    ret = (Ex, Ey, Ez)

    if magnetic_field:
        for component in (Hx, Hy, Hz):
            component *= phase
        Hx = np.squeeze(Hx)
        Hy = np.squeeze(Hy)
        Hz = np.squeeze(Hz)

        ret += (Hx, Hy, Hz)

    if return_grid:
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        X = np.squeeze(X)
        Y = np.squeeze(Y)
        Z = np.squeeze(Z)
        ret += (X, Y, Z)

    return ret


def fields_plane_wave(bead: Bead, x, y, z, theta=0, phi=0, polarization=(1, 0),
                      num_orders=None, return_grid=False,
                      total_field=True, magnetic_field=False,
                      verbose=False, grid=True
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

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    local_coordinates = LocalBeadCoordinates(
        x, y, z, bead.bead_diameter, grid=grid)
    internal_coordinates = InternalBeadCoordinates(local_coordinates)
    external_coordinates = ExternalBeadCoordinates(local_coordinates)

    an, _ = bead.ab_coeffs(num_orders)
    n_orders = an.size

    # Create a FarFieldData object that contains a single pixel == single plane
    # wave angles theta and phi and with amplitude and polarization (E_theta,
    # E_phi) given by `polarization`
    cos_theta = np.atleast_2d(np.cos(theta))
    sin_theta = np.atleast_2d(np.sin(theta))
    cos_phi = np.atleast_2d(np.cos(phi))
    sin_phi = np.atleast_2d(np.sin(phi))
    kz = bead.k * cos_theta
    kp = bead.k * sin_theta
    ky = -kp * sin_phi
    kx = -kp * cos_phi

    farfield_data = FarfieldData(
        Einf_theta=np.atleast_2d(polarization[0]) * kz,
        Einf_phi=np.atleast_2d(polarization[1]) * kz,
        aperture=np.atleast_2d(True),
        cos_theta=cos_theta, sin_theta=sin_theta,
        cos_phi=cos_phi, sin_phi=sin_phi,
        kz=kz, ky=ky, kx=kx, kp=kp
    )

    logging.info('Calculating Hankel functions and derivatives')
    external_radial_data = calculate_external(
        bead.k, external_coordinates.r, n_orders)

    logging.info(
        'Calculating Associated Legendre polynomials for external fields')
    legendre_data_ext = calculate_legendre(
        external_coordinates,
        farfield_data,
        n_orders
    )

    Ex = np.zeros(local_coordinates.coordinate_shape, dtype='complex128')
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)
    
    if magnetic_field:
        Hx = np.zeros_like(Ex)
        Hy = np.zeros_like(Ex)
        Hz = np.zeros_like(Ex)
    else:
        Hx = 0
        Hy = 0
        Hz = 0

    logging.info('Calculating external fields')

    calculate_fields(
        Ex, Ey, Ez, Hx, Hy, Hz,
        bead=bead, bead_center=(0, 0, 0), local_coordinates=external_coordinates,
        farfield_data=farfield_data, legendre_data=legendre_data_ext,
        external_radial_data=external_radial_data,
        internal=False,
        total_field=total_field, magnetic_field=magnetic_field,
    )

    logging.info('Calculating Bessel functions and derivatives')
    internal_radial_data = calculate_internal(
        bead.k1, internal_coordinates.r, n_orders
    )

    logging.info(
        'Calculating Associated Legendre polynomials for internal fields')
    legendre_data_int = calculate_legendre(
        internal_coordinates,
        farfield_data,
        n_orders
    )

    logging.info('Calculating internal fields')
    calculate_fields(
        Ex, Ey, Ez, Hx, Hy, Hz,
        bead=bead, bead_center=(0, 0, 0), local_coordinates=internal_coordinates,
        farfield_data=farfield_data, legendre_data=legendre_data_int,
        internal_radial_data=internal_radial_data,
        internal=True,
        total_field=total_field, magnetic_field=magnetic_field,
    )
    logging.getLogger().setLevel(loglevel)

    Ex = np.squeeze(Ex)
    Ey = np.squeeze(Ey)
    Ez = np.squeeze(Ez)

    ret = (Ex, Ey, Ez)

    if magnetic_field:
        Hx = np.squeeze(Hx)
        Hy = np.squeeze(Hy)
        Hz = np.squeeze(Hz)
        ret += (Hx, Hy, Hz)

    if return_grid:
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        X = np.squeeze(X)
        Y = np.squeeze(Y)
        Z = np.squeeze(Z)
        ret += (X, Y, Z)

    return ret


def forces_focus(
    f_input_field, objective,
    bead, bead_center=(0, 0, 0),
    bfp_sampling_n=31, num_orders=None, integration_orders=None,
    verbose=False
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

    an, _ = bead.ab_coeffs(num_orders)
    n_orders = an.size

    if integration_orders is None:
        # Go to the next higher available order of integration than should
        # be strictly necessary
        order = get_nearest_order(get_nearest_order(n_orders) + 1)
        x, y, z, w = get_integration_locations(order)
    else:
        x, y, z, w = get_integration_locations(
            get_nearest_order(np.amax((1, int(integration_orders))))
        )

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)

    xb = x * bead.bead_diameter * 0.51 + bead_center[0]
    yb = y * bead.bead_diameter * 0.51 + bead_center[1]
    zb = z * bead.bead_diameter * 0.51 + bead_center[2]

    Ex, Ey, Ez, Hx, Hy, Hz = fields_focus(
        f_input_field, objective, bead, bead_center, xb, yb, zb,
        bfp_sampling_n, num_orders, return_grid=False,
        total_field=True, magnetic_field=True, verbose=verbose,
        grid=False
    )
    _eps = EPS0 * bead.n_medium**2
    _mu = MU0

    Te11 = _eps * 0.5 * (np.abs(Ex)**2 - np.abs(Ey)**2 - np.abs(Ez)**2)
    Te12 = _eps * np.real(Ex * np.conj(Ey))
    Te13 = _eps * np.real(Ex * np.conj(Ez))
    Te22 = _eps * 0.5 * (np.abs(Ey)**2 - np.abs(Ex)**2 - np.abs(Ez)**2)
    Te23 = _eps * np.real(Ey * np.conj(Ez))
    Te33 = _eps * 0.5 * (np.abs(Ez)**2 - np.abs(Ey)**2 - np.abs(Ex)**2)
    Th11 = _mu * 0.5 * (np.abs(Hx)**2 - np.abs(Hy)**2 - np.abs(Hz)**2)
    Th12 = _mu * np.real(Hx * np.conj(Hy))
    Th13 = _mu * np.real(Hx * np.conj(Hz))
    Th22 = _mu * 0.5 * (np.abs(Hy)**2 - np.abs(Hx)**2 - np.abs(Hz)**2)
    Th23 = _mu * np.real(Hy * np.conj(Hz))
    Th33 = _mu * 0.5 * (np.abs(Hz)**2 - np.abs(Hy)**2 - np.abs(Hx)**2)
    F = np.zeros((3, 1))
    n = np.empty((3, 1))

    for k in np.arange(x.size):

        TE = np.asarray([
            [Te11[k], Te12[k], Te13[k]],
            [Te12[k], Te22[k], Te23[k]],
            [Te13[k], Te23[k], Te33[k]]])

        TH = np.asarray([
            [Th11[k], Th12[k], Th13[k]],
            [Th12[k], Th22[k], Th23[k]],
            [Th13[k], Th23[k], Th33[k]]])

        T = TE + TH
        n[0] = x[k]
        n[1] = y[k]
        n[2] = z[k]
        F += T @ n * w[k]
    # Note: factor 1/2 incorporated as 2 pi instead of 4 pi
    return np.squeeze(F) * (bead.bead_diameter * 0.51)**2 * 2 * np.pi


def absorbed_power_focus(
    f_input_field, objective,
    bead: Bead, bead_center=(0, 0, 0),
    bfp_sampling_n=31, num_orders=None, integration_orders=None,
    verbose=False
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

    an, _ = bead.ab_coeffs(num_orders)
    n_orders = an.size

    if integration_orders is None:
        # Go to the next higher available order of integration than should
        # be strictly necessary
        order = get_nearest_order(get_nearest_order(n_orders) + 1)
        x, y, z, w = get_integration_locations(order)
    else:
        x, y, z, w = get_integration_locations(get_nearest_order(
            np.amax((1, int(integration_orders)))
        ))

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)

    xb = x * bead.bead_diameter * 0.51 + bead_center[0]
    yb = y * bead.bead_diameter * 0.51 + bead_center[1]
    zb = z * bead.bead_diameter * 0.51 + bead_center[2]

    Ex, Ey, Ez, Hx, Hy, Hz = fields_focus(
        f_input_field, objective, bead, bead_center,
        xb, yb, zb, bfp_sampling_n, num_orders, return_grid=False,
        total_field=True, magnetic_field=True, verbose=verbose,
        grid=False
    )

    # Note: factor 1/2 incorporated later as 2 pi instead of 4 pi
    Px = (np.conj(Hz) * Ey - np.conj(Hy) * Ez).real
    Py = (np.conj(Hx) * Ez - np.conj(Hz) * Ex).real
    Pz = (np.conj(Hy) * Ex - np.conj(Hx) * Ey).real

    # Integral of - dot(P, n) over spherical surface
    Pabs = - np.sum((Px * x + Py * y + Pz * z) * w)

    return Pabs * (bead.bead_diameter * 0.51)**2 * 2 * np.pi


def scattered_power_focus(
    f_input_field, objective: Objective,
    bead: Bead, bead_center=(0, 0, 0),
    bfp_sampling_n=31, num_orders=None, integration_orders=None,
    verbose=False
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

    an, _ = bead.ab_coeffs(num_orders)
    n_orders = an.size

    if integration_orders is None:
        # Go to the next higher available order of integration than should
        # be strictly necessary
        order = get_nearest_order(get_nearest_order(n_orders) + 1)
        x, y, z, w = get_integration_locations(order)
    else:
        x, y, z, w = get_integration_locations(get_nearest_order(
            np.amax((1, int(integration_orders)))
        ))

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)

    xb = x * bead.bead_diameter * 0.51 + bead_center[0]
    yb = y * bead.bead_diameter * 0.51 + bead_center[1]
    zb = z * bead.bead_diameter * 0.51 + bead_center[2]

    Ex, Ey, Ez, Hx, Hy, Hz = fields_focus(
        f_input_field, objective,
        bead, bead_center,
        xb, yb, zb, bfp_sampling_n, num_orders, return_grid=False,
        total_field=False, magnetic_field=True, verbose=verbose,
        grid=False
    )

    # Note: factor 1/2 incorporated later as 2 pi instead of 4 pi
    Px = (np.conj(Hz) * Ey - np.conj(Hy) * Ez).real
    Py = (np.conj(Hx) * Ez - np.conj(Hz) * Ex).real
    Pz = (np.conj(Hy) * Ex - np.conj(Hx) * Ey).real

    # integral of dot(P, n) over spherical surface:
    Psca = np.sum((Px * x + Py * y + Pz * z) * w)

    return Psca * (bead.bead_diameter * 0.51)**2 * 2 * np.pi

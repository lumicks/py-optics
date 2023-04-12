from .bead import Bead
from .lebedev_laikov import *
from .local_coordinates import LocalBeadCoordinates
from .legendre import *
from .hankel import *
from .objective import *

import numpy as np
from numba import njit
from scipy.constants import (
    speed_of_light as _C,
    epsilon_0 as _EPS0,
    mu_0 as _MU0
)
import logging


def _calculate_fields(
    Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray,
    Hx: np.ndarray, Hy: np.ndarray, Hz: np.ndarray,
    bead: Bead = None,
    bead_center: tuple = (0, 0, 0),
    local_coordinates: LocalBeadCoordinates = None,
    farfield_data: FarfieldData = None,
    legendre_data: AssociatedLegendreData = None,
    external_radial_data: ExternalRadialData = None,
    internal_radial_data: InternalRadialData = None,
    internal: bool = False,
    total_field: bool = True,
    magnetic_field: bool = False,
):
    """Calculate the internal & external field from precalculated,
    but compressed, Legendre polynomials
    Compromise between speed and memory use"""
    if internal:
        r = local_coordinates.r_inside
        local_coords = local_coordinates.xyz_stacked(inside=True)
        region = np.atleast_1d(np.squeeze(local_coordinates.region_inside_bead))
        n_orders = internal_radial_data.sphBessel.shape[0]
    else:
        r = local_coordinates.r_outside
        local_coords = local_coordinates.xyz_stacked(inside=False)
        region = np.atleast_1d(np.squeeze(local_coordinates.region_outside_bead))
        n_orders = external_radial_data.krH.shape[0]
    if r.size == 0:
        return

    E = np.empty((3, local_coordinates.r.shape[1]), dtype='complex128')
    if magnetic_field:
        H = np.empty(E.shape, dtype='complex128')
    
    # preallocate memory for expanded legendre derivatives
    alp_expanded = np.empty((n_orders, r.size))
    alp_sin_expanded = np.empty(alp_expanded.shape)
    alp_deriv_expanded = np.empty(alp_expanded.shape)

    an, bn = bead.ab_coeffs(n_orders)
    cn, dn = bead.cd_coeffs(n_orders)

    n_pupil_samples = farfield_data.Einf_phi.shape[0]
    cosT = np.empty(r.shape)
    for m in range(n_pupil_samples):
        for p in range(n_pupil_samples):
            if not farfield_data.aperture[p, m]:  # Skip points outside aperture
                continue
            matrices = [
                _R_th(farfield_data.cos_theta[p,m], farfield_data.sin_theta[p,m]) @ 
                _R_phi(farfield_data.cos_phi[p,m], -farfield_data.sin_phi[p,m]),
                _R_phi(0, -1) @ 
                _R_th(farfield_data.cos_theta[p,m], farfield_data.sin_theta[p,m]) @ 
                _R_phi(farfield_data.cos_phi[p,m], -farfield_data.sin_phi[p,m])
            ]
            
            E0 = [farfield_data.Einf_theta[p, m], farfield_data.Einf_phi[p, m]]
        
            for polarization in range(2):
                A = matrices[polarization]
                coords = A @ local_coords
                x = coords[0,:]
                y = coords[1,:]
                z = coords[2,:]

                if polarization == 0:
                    if internal:
                        cosT[r > 0] = z[r > 0] / r[r > 0]
                        cosT[r == 0] = 1
                    else:
                        cosT[:] = z / r
                    cosT[cosT > 1] = 1
                    cosT[cosT < -1] = -1
                
                    # Expand the legendre derivatives from the unique version of
                    # cos(theta)
                    alp_expanded[:] = legendre_data.associated_legendre(p, m)
                    alp_sin_expanded[:] = legendre_data.associated_legendre_over_sin_theta(p, m)
                    alp_deriv_expanded[:] = legendre_data.associated_legendre_dtheta(p, m)
                
                rho_l = np.hypot(x, y)
                cosP = np.empty(rho_l.shape)
                sinP = np.empty(rho_l.shape)
                where = rho_l > 0
                cosP[where] = x[where]/rho_l[where]
                sinP[where] = y[where]/rho_l[where]
                cosP[rho_l == 0] = 1
                sinP[rho_l == 0] = 0
                
                E[:] = 0
                if internal:
                    E[:, region] = _internal_field_fixed_r(
                        cn, dn, 
                        internal_radial_data.sphBessel,
                        internal_radial_data.jn_over_k1r, 
                        internal_radial_data.jn_1, 
                        alp_expanded,
                        alp_sin_expanded, alp_deriv_expanded,
                        cosT, cosP, sinP
                    )
                else:
                    E[:, region] = _scattered_field_fixed_r(
                        an, bn,
                        external_radial_data.krH,
                        external_radial_data.dkrH_dkr,
                        external_radial_data.k0r,
                        alp_expanded, alp_sin_expanded,
                        alp_deriv_expanded, cosT, cosP, sinP,
                        total_field
                    )

                E = np.matmul(A.T, E)
                E[:, region] *= E0[polarization] * np.exp(1j * (
                    farfield_data.Kx[p,m] * bead_center[0] +
                    farfield_data.Ky[p,m] * bead_center[1] +
                    farfield_data.Kz[p,m] * bead_center[2])
                )

                Ex[:,:,:] += np.reshape(E[0,:], local_coordinates.coordinate_shape)
                Ey[:,:,:] += np.reshape(E[1,:], local_coordinates.coordinate_shape)
                Ez[:,:,:] += np.reshape(E[2,:], local_coordinates.coordinate_shape)

                if magnetic_field:
                    H[:] = 0
                    if internal:
                        H[:, region] = _internal_H_field_fixed_r(
                            cn, dn,
                            internal_radial_data.sphBessel,
                            internal_radial_data.jn_over_k1r,
                            internal_radial_data.jn_1,
                            alp_expanded,
                            alp_sin_expanded, alp_deriv_expanded,
                            cosT, cosP, sinP, bead.n_bead
                        )
                    else:
                        H[:, region] = _scattered_H_field_fixed_r(
                            an, bn,
                            external_radial_data.krH,
                            external_radial_data.dkrH_dkr,
                            external_radial_data.k0r,
                            alp_expanded, alp_sin_expanded,
                            alp_deriv_expanded, cosT, cosP, sinP, 
                            bead.n_medium, total_field
                        )

                    H = np.matmul(A.T, H)
                    H[:, region] *= E0[polarization] * np.exp(1j * (
                        farfield_data.Kx[p,m] * bead_center[0] +
                        farfield_data.Ky[p,m] * bead_center[1] +
                        farfield_data.Kz[p,m] * bead_center[2])
                    )

                    Hx[:,:,:] += np.reshape(H[0,:], local_coordinates.coordinate_shape)
                    Hy[:,:,:] += np.reshape(H[1,:], local_coordinates.coordinate_shape)
                    Hz[:,:,:] += np.reshape(H[2,:], local_coordinates.coordinate_shape)


@njit(cache=True)
def _scattered_field_fixed_r(an: np.ndarray, bn: np.ndarray, krh: np.ndarray, 
                            dkrh_dkr: np.ndarray, k0r: np.ndarray, 
                            alp: np.ndarray, alp_sin: np.ndarray,
                            alp_deriv: np.ndarray,
                            cos_theta: np.ndarray, cosP: np.ndarray,
                            sinP: np.ndarray,
                            total_field=True):

    # Radial, theta and phi-oriented fields
    Er = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
    Et = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
    Ep = np.zeros((1,cos_theta.shape[0]), dtype='complex128')

    sinT = np.sqrt(1 - cos_theta**2)  # for theta = 0...pi

    L = np.arange(start=1, stop=an.size + 1)
    C1 = 1j**(L + 1) * (2 * L + 1)
    C2 = C1 / (L * (L + 1))
    for L in range(an.size, 0, -1):
        Er += C1[L-1] * an[L - 1] * krh[L - 1,:] * alp[L-1,:]

        Et += C2[L-1] * (an[L - 1] *
                dkrh_dkr[L - 1,:] * alp_deriv[L - 1,:] + 1j*bn[L - 1] *
                krh[L - 1,:] * alp_sin[L - 1,:])

        Ep += C2[L-1] * (an[L - 1] *
            dkrh_dkr[L - 1,:] * alp_sin[L - 1,:] + 1j * bn[L - 1] *
            krh[L - 1,:] * alp_deriv[L - 1,:])

    Er *= -cosP / (k0r)**2
    Et *= -cosP / (k0r)
    Ep *= sinP / (k0r)
    # Cartesian components
    Ex = Er * sinT * cosP + Et * cos_theta * cosP - Ep * sinP
    Ey = Er * sinT * sinP + Et * cos_theta * sinP + Ep * cosP
    Ez = Er * cos_theta - Et * sinT
    if total_field:
        # Incident field (x-polarized)
        Ei = np.exp(1j * k0r * cos_theta)
        return np.concatenate((Ex + Ei, Ey, Ez), axis=0)
    else:
        return np.concatenate((Ex, Ey, Ez), axis=0)


@njit(cache=True)
def _internal_field_fixed_r(cn: np.ndarray, dn: np.ndarray, 
    sphBessel: np.ndarray, jn_over_k1r: np.ndarray, jn_1: np.ndarray, 
    alp: np.ndarray, alp_sin: np.ndarray, alp_deriv: np.ndarray,
    cos_theta: np.ndarray, cosP: np.ndarray, sinP: np.ndarray):

    # Radial, theta and phi-oriented fields
    Er = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
    Et = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
    Ep = np.zeros((1,cos_theta.shape[0]), dtype='complex128')

    sinT = np.sqrt(1 - cos_theta**2)
    
    for n in range(cn.size, 0, -1):
        Er += - (1j**(n + 1) * (2*n + 1)  * alp[n - 1, :] * dn[n - 1] *
                jn_over_k1r[n - 1, :])

        Et += 1j**n * (2 * n + 1) / (n * (n + 1)) * (cn[n - 1] *
                alp_sin[n - 1, :] * sphBessel[n - 1, :] - 1j * dn[n - 1] *
                alp_deriv[n - 1, :] * (jn_1[n - 1, :] -
                                    n * jn_over_k1r[n - 1, :]))

        Ep += - 1j**n * (2 * n + 1) / (n * (n + 1)) * (cn[n - 1] *
            alp_deriv[n - 1, :] * sphBessel[n - 1, :] -
            1j*dn[n - 1] * alp_sin[n - 1, :] * (jn_1[n - 1, :] - n *
                                                jn_over_k1r[n - 1, :]))


    Er *= -cosP
    Et *= -cosP
    Ep *= -sinP
    # Cartesian components
    Ex = Er * sinT * cosP + Et * cos_theta * cosP - Ep * sinP
    Ey = Er * sinT * sinP + Et * cos_theta * sinP + Ep * cosP
    Ez = Er * cos_theta - Et * sinT
    return np.concatenate((Ex, Ey, Ez), axis=0)


@njit(cache=True)
def _scattered_H_field_fixed_r(an: np.ndarray, bn: np.ndarray, krh: np.ndarray, 
                            dkrh_dkr: np.ndarray, k0r: np.ndarray, 
                            alp: np.ndarray, alp_sin: np.ndarray,
                            alp_deriv: np.ndarray,
                            cos_theta: np.ndarray, cosP: np.ndarray,
                            sinP: np.ndarray, n_medium: float,
                            total_field=True):
    
    # Radial, theta and phi-oriented fields
    Hr = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
    Ht = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
    Hp = np.zeros((1,cos_theta.shape[0]), dtype='complex128')

    sinT = np.sqrt(1 - cos_theta**2)

    L = np.arange(start=1, stop=an.size + 1)
    C1 = 1j**L * (2 * L + 1)
    C2 = C1 / (L * (L + 1))
    for L in range(an.size, 0, -1):
        Hr += C1[L-1] * 1j * bn[L - 1] * krh[L - 1,:] * alp[L-1,:]

        Ht += C2[L-1] * (1j* bn[L - 1] *
                dkrh_dkr[L - 1,:] * alp_deriv[L - 1,:] - an[L - 1] *
                krh[L - 1,:] * alp_sin[L - 1,:])

        Hp += C2[L-1] * (1j * bn[L - 1] *
            dkrh_dkr[L - 1,:] * alp_sin[L - 1,:] - an[L - 1] *
            krh[L - 1,:] * alp_deriv[L - 1,:])
    
    # Extra factor of -1 as B&H does not include the Condon–Shortley phase, 
    # but our associated Legendre polynomials do include it
    Hr *= -sinP / (k0r)**2 * n_medium / (_C * _MU0)
    Ht *= -sinP / (k0r) * n_medium / (_C * _MU0)
    Hp *= -cosP / (k0r) * n_medium / (_C * _MU0)
    
    # Cartesian components
    Hx = Hr * sinT * cosP + Ht * cos_theta * cosP - Hp * sinP
    Hy = Hr * sinT * sinP + Ht * cos_theta * sinP + Hp * cosP
    Hz = Hr * cos_theta - Ht * sinT
    if total_field:
        # Incident field (E field x-polarized)
        Hi = np.exp(1j * k0r * cos_theta) * n_medium / (_C * _MU0)
        return np.concatenate((Hx, Hy + Hi, Hz), axis=0)
    else:
        return np.concatenate((Hx, Hy, Hz), axis=0)


@njit(cache=True)
def _internal_H_field_fixed_r(cn: np.ndarray, dn: np.ndarray, 
    sphBessel: np.ndarray, jn_over_k1r: np.ndarray, jn_1: np.ndarray, 
    alp: np.ndarray, alp_sin: np.ndarray, alp_deriv: np.ndarray,
    cos_theta: np.ndarray, cosP: np.ndarray, sinP: np.ndarray, 
    n_bead: np.complex128):

    # Radial, theta and phi-oriented fields
    Hr = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
    Ht = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
    Hp = np.zeros((1,cos_theta.shape[0]), dtype='complex128')

    sinT = np.sqrt(1 - cos_theta**2)

    for n in range(cn.size, 0, -1):
        Hr += (1j**(n + 1) * (2*n + 1)  * alp[n - 1, :] * cn[n - 1] *
                jn_over_k1r[n - 1, :])

        Ht += 1j**n * (2 * n + 1) / (n * (n + 1)) * (dn[n - 1] *
                alp_sin[n - 1, :] * sphBessel[n - 1, :] - 1j * cn[n - 1] *
                alp_deriv[n - 1, :] * (jn_1[n - 1, :] -
                                    n * jn_over_k1r[n - 1, :]))

        Hp += - 1j**n * (2 * n + 1) / (n * (n + 1)) * (dn[n - 1] *
            alp_deriv[n - 1, :] * sphBessel[n - 1, :] -
            1j*cn[n - 1] * alp_sin[n - 1, :] * (jn_1[n - 1, :] - n *
                                                jn_over_k1r[n - 1, :]))


    Hr *= sinP * n_bead / (_C * _MU0)
    Ht *= -sinP * n_bead / (_C * _MU0)
    Hp *= cosP * n_bead / (_C * _MU0)
    # Cartesian components
    Hx = Hr * sinT * cosP + Ht * cos_theta * cosP - Hp * sinP
    Hy = Hr * sinT * sinP + Ht * cos_theta * sinP + Hp * cosP
    Hz = Hr * cos_theta - Ht * sinT
    return np.concatenate((Hx, Hy, Hz), axis=0)


def fields_gaussian_focus(
    beam_power: float, 
    objective: Objective, filling_factor=0.9,
    bead: Bead = None, bead_center=(0,0,0),
    x=0, y=0, z=0,
    bfp_sampling_n=31, num_orders=None,
    return_grid=False,  total_field=True,
    inside_bead=True, H_field=False, verbose=False, 
    grid=True
):
    """Calculate the three-dimensional electromagnetic field of a bead the
    focus of a of a Gaussian beam, using the angular spectrum of plane waves
    and Mie theory. 

    This function correctly incorporates the polarized nature of light in a
    focus. In other words, the polarization state at the focus includes
    electric fields in the x, y, and z directions. The input is taken to be
    polarized along the x direction.

    Parameters
    ---------- 
    beam_power : power of the laser beam before entering the objective, in Watt.
    filling_factor : filling factor of the Gaussian beam over the
        aperture, defined as w0/R. Here, w0 is the waist of the Gaussian
        beam and R is the radius of the aperture. Range 0...Inf 
    objective : instance of the Objective class.
    x : array of x locations for evaluation, in meters 
    y : array of y locations for evaluation, in meters 
    z : array of z locations for evaluation, in meters
    bead_center : tuple: tuple of three floating point numbers
        determining the x, y and z position of the bead center in 3D space,
        in meters
    bfp_sampling_n : (Default value = 31) Number of discrete steps with
        which the back focal plane is sampled, from the center to the edge.
        The total number of plane waves scales with the square of
        bfp_sampling_n 
    num_orders : number of order that should be included in the
        calculation the Mie solution. If it is None (default), the code will
        use the number_of_orders() method to calculate a sufficient number.
    return_grid: (Default value = False) return the sampling grid in the
        matrices X, Y and Z
    total_field : If True, return the total field of incident and
        scattered electromagnetic field (default). If False, then only
        return the scattered field outside the bead. Inside the bead, the
        full field is always returned.
    inside_bead : If True (default), return the fields inside the bead. If
        False, do not calculate the fields inside the bead. Zero is returned
        for positions inside the bead instead. 
    H_field : If True, return the magnetic fields as well. If false
        (default), do not return the magnetic fields.
    verbose : If True, print statements on the progress of the calculation.
        Default is False
    grid: If True (default), interpret the vectors or scalars x, y and z as
        the input for the numpy.meshgrid function, and calculate the fields
        at the locations that are the result of the numpy.meshgrid output.
        If False, interpret the x, y and z vectors as the exact locations
        where the field needs to be evaluated. In that case, all vectors
        need to be of the same length.

    Returns
    -------
    Ex : the electric field along x, as a function of (x, y, z) 
    Ey : the electric field along y, as a function of (x, y, z) 
    Ez : the electric
    field along z, as a function of (x, y, z) 
    Hx : the magnetic field along x, as a function of (x, y, z) 
    Hy : the magnetic field along y, as a function of (x, y, z) 
    Hz : the magnetic field along z, as a function of (x, y, z) 
        These values are only returned when H_field is True
    X : x coordinates of the sampling grid
    Y : y coordinates of the sampling grid
    Z : z coordinates of the sampling grid
        These values are only returned if return_grid is True
    """

    w0 = filling_factor * objective.focal_length * objective.NA / bead.n_medium  # [m]
    I0 = 2 * beam_power / (np.pi * w0**2)  # [W/m^2]
    E0 = (I0 * 2/(_EPS0 * _C * objective.n_bfp))**0.5  # [V/m]

    def gaussian_beam(X_bfp, Y_bfp, **kwargs): 
        Ex = np.exp(-(X_bfp**2 + Y_bfp**2) / w0**2) * E0
        return (Ex, None)
    
    return fields_focus(
        gaussian_beam, objective, bead,
        x=x, y=y, z=z, 
        bead_center=bead_center, bfp_sampling_n=bfp_sampling_n,
        num_orders=num_orders, return_grid=return_grid, 
        total_field=total_field, inside_bead=inside_bead, H_field=H_field,
        verbose=verbose, grid=grid
    )

def fields_focus(f_input_field, objective: Objective,
            bead: Bead, bead_center=(0,0,0), 
            x=0, y=0, z=0, 
            bfp_sampling_n=31, num_orders=None,
            return_grid=False,  total_field=True,
            inside_bead=True, magnetic_field=False, 
            verbose=False, grid=True
            ):
    """Calculate the three-dimensional electromagnetic field of a bead in the
    focus of an arbitrary input beam, going through an objective with a certain NA and focal
    length. Implemented with the angular spectrum of plane waves and Mie theory. 

    This function correctly incorporates the polarized nature of light in a focus. In other
    words, the polarization state at the focus includes electric fields in the x, y, and z
    directions. The input can be a combination of x- and y-polarized light of complex
    amplitudes.

    Parameters
    ----------
    f_input_field : function with signature f(X_BFP, Y_BFP, R, Rmax), where
        X_BFP is a grid of x locations in the back focal plane, determined by the focal length
        and NA of the objective. Y_BFP is the corresponding grid of y locations, and R is the
        radial distance from the center of the back focal plane. Rmax is the largest distance
        that falls inside the NA, but R will contain larger numbers as the back focal plane is
        sampled with a square grid. 
        The function must return a tuple (Ex, Ey), which are the electric fields in
        the x- and y- direction, respectively, at the sample locations in the back focal plane.
        The fields may be complex, so a phase difference between x and y is possible. If only
        one polarization is used, the other return value must be None, e.g., y polarization
        would return (None, Ey). The fields are post-processed such that any part that
        falls outside of the NA is set to zero.
    objective : instance of the Objective class
    bead : instance of the Bead class
    x : array of x locations for evaluation, in meters 
    y : array of y locations for evaluation, in meters 
    z : array of z locations for evaluation, in meters
    bead_center : tuple of three floating point numbers determining
        the x, y and z position of the bead center in 3D space, in meters
    bfp_sampling_n : (Default value = 31) Number of discrete steps with which
        the back focal plane is sampled, from the center to the edge. The total number of plane
        waves scales with the square of bfp_sampling_n 
    num_orders: number of order that should be included in the calculation
        the Mie solution. If it is None (default), the code will use the number_of_orders()
        method to calculate a sufficient number.
    return_grid : (Default value = False) return the sampling grid in the
        matrices X, Y and Z
    total_field : If True, return the total field of incident and scattered
        electromagnetic field (default). If False, then only return the scattered field outside
        the bead. Inside the bead, the full field is always returned.
    inside_bead : If True (default), return the fields inside the bead. If
        False, do not calculate the fields inside the bead. Zero is returned for positions
        inside the bead instead. 
    H_field: If True, return the magnetic fields as well. If false
        (default), do not return the magnetic fields.
    verbose: If True, print statements on the progress of the calculation.
        Default is False
    grid: If True (default), interpret the vectors or scalars x, y and z as
        the input for the numpy.meshgrid function, and calculate the fields at the locations
        that are the result of the numpy.meshgrid output. If False, interpret the x, y and z
        vectors as the exact locations where the field needs to be evaluated. In that case, all
        vectors need to be of the same length.

    Returns
    -------
    Ex : the electric field along x, as a function of (x, y, z) 
    Ey : the electric field along y, as a function of (x, y, z) 
    Ez : the electric field along z, as a function of (x, y, z) 
    Hx : the magnetic field along x, as a function of (x, y, z) 
    Hy : the magnetic field along y, as a function of (x, y, z) 
    Hz : the magnetic field along z, as a function of (x, y, z) 
        These values are only returned when H_field is True
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
    # TODO: warning criteria for undersampling/aliasing
    #M = int(np.max((31, 2 * NA**2 * np.max(np.abs(z)) /
    #        (np.sqrt(self.n_medium**2 - NA**2) * self.lambda_vac))))
    #if M > bfp_sampling_n:
    #    print('bfp_sampling_n lower than recommendation for convergence')

    local_coordinates = LocalBeadCoordinates(
        x, y, z, bead.bead_diameter, bead_center, grid=grid
    )

    an, _ = bead.ab_coeffs(num_orders)
    n_orders = an.size

    bfp_coords, bfp_fields = sample_bfp(
        f_input_field=f_input_field, bfp_sampling_n=bfp_sampling_n, objective=objective
    )

    farfield_data = bfp_to_farfield(bfp_coords, bfp_fields, objective, bead.lambda_vac)
    
    logging.info('Calculating Hankel functions and derivatives')
    external_radial_data = calculate_external(bead.k, local_coordinates.r_outside, n_orders)

    logging.info('Calculating Associated Legendre polynomials for external fields')
    legendre_data_ext = calculate_legendre(
        local_coordinates.xyz_stacked(inside=False),
        local_coordinates.r_outside,
        bfp_coords.aperture,
        farfield_data.cos_theta,
        farfield_data.sin_theta,
        farfield_data.cos_phi,
        farfield_data.sin_phi,
        n_orders
    )

    Ex = np.zeros(local_coordinates.coordinate_shape, dtype='complex128')
    Ey = np.zeros(local_coordinates.coordinate_shape, dtype='complex128')
    Ez = np.zeros(local_coordinates.coordinate_shape, dtype='complex128')
    if magnetic_field:
        Hx = np.zeros(local_coordinates.coordinate_shape, dtype='complex128')
        Hy = np.zeros(local_coordinates.coordinate_shape, dtype='complex128')
        Hz = np.zeros(local_coordinates.coordinate_shape, dtype='complex128')
    else:
        Hx = 0
        Hy = 0
        Hz = 0
        
    logging.info('Calculating external fields')

    _calculate_fields(
        Ex, Ey, Ez, Hx, Hy, Hz,
        bead=bead, bead_center=bead_center, local_coordinates=local_coordinates,
        farfield_data=farfield_data, legendre_data=legendre_data_ext,
        external_radial_data=external_radial_data,
        internal=False,
        total_field=total_field, magnetic_field=magnetic_field,
    )

    if inside_bead:
        
        logging.info('Calculating Bessel functions and derivatives')
        internal_radial_data = calculate_internal(
            bead.k1, local_coordinates.r_inside, n_orders
        )

        logging.info('Calculating Associated Legendre polynomials for internal fields')
        legendre_data_int = calculate_legendre(
            local_coordinates.xyz_stacked(inside=True),
            local_coordinates.r_inside,
            bfp_coords.aperture,
            farfield_data.cos_theta,
            farfield_data.sin_theta,
            farfield_data.cos_phi,
            farfield_data.sin_phi,
            n_orders
        )

        logging.info('Calculating internal fields')
        _calculate_fields(
            Ex, Ey, Ez, Hx, Hy, Hz,
            bead=bead, bead_center=bead_center, local_coordinates=local_coordinates,
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
    Ex *= phase
    Ey *= phase
    Ez *= phase

    Ex = np.squeeze(Ex)
    Ey = np.squeeze(Ey)
    Ez = np.squeeze(Ez)

    if return_grid:
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        X = np.squeeze(X)
        Y = np.squeeze(Y)
        Z = np.squeeze(Z)

    if magnetic_field:
        Hx *= phase
        Hy *= phase
        Hz *= phase
        Hx = np.squeeze(Hx)
        Hy = np.squeeze(Hy)
        Hz = np.squeeze(Hz)

        if return_grid:
            return Ex, Ey, Ez, Hx, Hy, Hz, X, Y, Z
        else:
            return Ex, Ey, Ez, Hx, Hy, Hz

    if return_grid:
        return Ex, Ey, Ez, X, Y, Z
    else:
        return Ex, Ey, Ez

def fields_plane_wave(self, x, y, z, theta=0, phi=0, polarization=(1,0), 
                        num_orders=None, return_grid=False, 
                        total_field=True, inside_bead=True, H_field=False,
                        verbose=False, grid=True
                        ):
    """Calculate the electromagnetic field of a bead, subject to excitation
    by a plane wave. The plane wave can be at an angle theta and phi, and
    have a polarization state that is the combination of the (complex)
    amplitudes of a theta-polarization state and phi-polarized state. If
    theta == 0 and phi == 0, the theta polarization points along +x axis and
    the wave travels into the +z direction.
    
    Parameters
    ----------
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
    inside_bead : If True (default), return the fields inside the bead. If
        False, do not calculate the fields inside the bead. Zero is returned
        for positions inside the bead instead. 
    H_field : If True, return the magnetic fields as well. If false
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
    Ez : the electric
    field along z, as a function of (x, y, z) 
    Hx : the magnetic field along x, as a function of (x, y, z) 
    Hy : the magnetic field along y, as a function of (x, y, z) 
    Hz : the magnetic field along z, as a function of (x, y, z) 
        These values are only returned when H_field is True
    X : x coordinates of the sampling grid
    Y : y coordinates of the sampling grid
    Z : z coordinates of the sampling grid
        These values are only returned if return_grid is True
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    self._init_local_coordinates(x,y,z, (0,0,0), grid=grid)
    self._get_mie_coefficients(num_orders)

    # Abuse the back focal plane to contain a single pixel/plane wave, at 
    # angles theta and phi and with amplitude and polarization 
    # (E_theta, E_phi) given by `polarization`
    self._Einf_theta = np.atleast_2d(polarization[0])
    self._Einf_phi = np.atleast_2d(polarization[1])
    self._aperture = np.atleast_2d(False)
    #self._Einfx = np.ones(self._Einf_phi.shape)
    self._cosT = np.atleast_2d(np.cos(theta))
    self._sinT = np.atleast_2d(np.sin(theta))
    self._cosP = np.atleast_2d(np.cos(phi))
    self._sinP = np.atleast_2d(np.sin(phi))
    self._Kz = np.zeros(self._Einf_phi.shape)
    self._Ky = np.zeros(self._Einf_phi.shape)
    self._Kx = np.zeros(self._Einf_phi.shape)
    
    if verbose:
        print('Hankel functions')
    self._init_hankel()
    if verbose:
        print('Legendre functions')
    self._init_legendre(outside=True)
    
    Ex = np.zeros(self._local_coordinates.coordinate_shape, dtype='complex128')
    Ey = np.zeros(self._local_coordinates.coordinate_shape, dtype='complex128')
    Ez = np.zeros(self._local_coordinates.coordinate_shape, dtype='complex128')

    if H_field:
        Hx = np.zeros(self._local_coordinates.coordinate_shape, dtype='complex128')
        Hy = np.zeros(self._local_coordinates.coordinate_shape, dtype='complex128')
        Hz = np.zeros(self._local_coordinates.coordinate_shape, dtype='complex128')
    else:
        Hx = 0
        Hy = 0
        Hz = 0
    
    if verbose:
        print('External field')
    self._calculate_field_speed_mem(Ex, Ey, Ez, Hx, Hy, Hz, False, 
        total_field, H_field, bead_center=(0,0,0))

    if inside_bead:
        if verbose:
            print('Bessel functions')
        self._init_bessel()
        if verbose:
            print('Legendre functions')
        self._init_legendre(outside=False)
        if verbose:
            print('Internal field')
        self._calculate_field_speed_mem(Ex, Ey, Ez, Hx, Hy, Hz, True, 
            total_field, H_field, bead_center=(0,0,0))

    Ex = np.squeeze(Ex)
    Ey = np.squeeze(Ey)
    Ez = np.squeeze(Ez)

    if return_grid:
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        X = np.squeeze(X)
        Y = np.squeeze(Y)
        Z = np.squeeze(Z)

    if H_field:
        Hx = np.squeeze(Hx)
        Hy = np.squeeze(Hy)
        Hz = np.squeeze(Hz)
        
        if return_grid:
            return Ex, Ey, Ez, Hx, Hy, Hz, X, Y, Z
        else:
            return Ex, Ey, Ez, Hx, Hy, Hz

    if return_grid:
        return Ex, Ey, Ez, X, Y, Z
    else:
        return Ex, Ey, Ez


def forces_focused_fields(self, f_input_field, n_bfp=1.0,
            focal_length=4.43e-3, NA=1.2,
            bead_center=(0,0,0),
            bfp_sampling_n=31, num_orders=None, integration_orders=None,
            verbose=False
            ):
    """Calculate the forces on a bead in the focus of an arbitrary input
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
    n_bfp : refractive index at the back focal plane of the objective
        focused focal_length: focal length of the objective, in meters 
    focal_length : focal length of the objective, in meters
    NA : Numerical Aperture n_medium * sin(theta_max) of the objective
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

    self._get_mie_coefficients(num_orders)
    if integration_orders is None:
        # Go to the next higher available order of integration than should 
        # be strictly necessary
        order = get_nearest_order(get_nearest_order(self._n_coeffs) + 1)
        x, y, z, w = get_integration_locations(order)
    else:
        x, y, z, w = get_integration_locations(get_nearest_order(
            np.amax((1,int(integration_orders)))
            ))

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)
    
    xb = x * self.bead_diameter * 0.51 + bead_center[0]
    yb = y * self.bead_diameter * 0.51 + bead_center[1]
    zb = z * self.bead_diameter * 0.51 + bead_center[2]

    Ex, Ey, Ez, Hx, Hy, Hz = self.fields_focus(
        f_input_field, n_bfp, focal_length, NA, xb, yb, zb,
        bead_center, bfp_sampling_n, num_orders, return_grid=False,
        total_field=True, inside_bead=False, H_field=True, verbose=verbose,
        grid=False
    )
    _eps = _EPS0 * self.n_medium**2
    _mu = _MU0

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
            [ Te11[k], Te12[k], Te13[k]],
            [ Te12[k], Te22[k], Te23[k]],
            [ Te13[k], Te23[k], Te33[k]]])

        TH = np.asarray([
            [ Th11[k], Th12[k], Th13[k]],
            [ Th12[k], Th22[k], Th23[k]],
            [ Th13[k], Th23[k], Th33[k]]])

        T = TE + TH
        n[0] = x[k]
        n[1] = y[k]
        n[2] = z[k]
        F += T @ n * w[k]
    # Note: factor 1/2 incorporated as 2 pi instead of 4 pi
    return F * (self.bead_diameter * 0.51)**2 * 2 * np.pi



def absorbed_power_focus(self, f_input_field, n_bfp=1.0,
                focal_length=4.43e-3, NA=1.2,
                bead_center=(0,0,0),
                bfp_sampling_n=31, num_orders=None, integration_orders=None,
                verbose=False
                ):
        """Calculate the dissipated power in a bead in the focus of an arbitrary input
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
        n_bfp : refractive index at the back focal plane of the objective
            focused focal_length: focal length of the objective, in meters 
        focal_length : focal length of the objective, in meters
        NA : Numerical Aperture n_medium * sin(theta_max) of the objective
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
        Pabs : the absorbed power in Watts.
        """

        self._get_mie_coefficients(num_orders)
        if integration_orders is None:
            # Go to the next higher available order of integration than should 
            # be strictly necessary
            order = get_nearest_order(get_nearest_order(self._n_coeffs) + 1)
            x, y, z, w = get_integration_locations(order)
        else:
            x, y, z, w = get_integration_locations(get_nearest_order(
                np.amax((1,int(integration_orders)))
                ))

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        w = np.asarray(w)
    
        xb = x * self.bead_diameter * 0.51 + bead_center[0]
        yb = y * self.bead_diameter * 0.51 + bead_center[1]
        zb = z * self.bead_diameter * 0.51 + bead_center[2]

        Ex, Ey, Ez, Hx, Hy, Hz = self.fields_focus(
            f_input_field, n_bfp, focal_length, NA, xb, yb, zb,
            bead_center, bfp_sampling_n, num_orders, return_grid=False,
            total_field=True, inside_bead=False, H_field=True, verbose=verbose,
            grid=False
        )
    
        # Note: factor 1/2 incorporated later as 2 pi instead of 4 pi
        Px = (np.conj(Hz)*Ey - np.conj(Hy)*Ez).real
        Py = (np.conj(Hx)*Ez - np.conj(Hz)*Ex).real
        Pz = (np.conj(Hy)*Ex - np.conj(Hx)*Ey).real
        Pabs = np.sum((Px * x + Py * y + Pz * z) * w)

        return Pabs * (self.bead_diameter * 0.51)**2 * 2 * np.pi


def scattered_power_focus(self, f_input_field, n_bfp=1.0,
                focal_length=4.43e-3, NA=1.2,
                bead_center=(0,0,0),
                bfp_sampling_n=31, num_orders=None, integration_orders=None,
                verbose=False
                ):
        """Calculate the scattered power by a bead in the focus of an arbitrary input
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
        n_bfp : refractive index at the back focal plane of the objective
            focused focal_length: focal length of the objective, in meters 
        focal_length : focal length of the objective, in meters
        NA : Numerical Aperture n_medium * sin(theta_max) of the objective
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
        Psca : the scattered power in Watts.
        """

        self._get_mie_coefficients(num_orders)
        if integration_orders is None:
            # Go to the next higher available order of integration than should 
            # be strictly necessary
            order = get_nearest_order(get_nearest_order(self._n_coeffs) + 1)
            x, y, z, w = get_integration_locations(order)
        else:
            x, y, z, w = get_integration_locations(get_nearest_order(
                np.amax((1,int(integration_orders)))
                ))

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        w = np.asarray(w)
    
        xb = x * self.bead_diameter * 0.51 + bead_center[0]
        yb = y * self.bead_diameter * 0.51 + bead_center[1]
        zb = z * self.bead_diameter * 0.51 + bead_center[2]

        Ex, Ey, Ez, Hx, Hy, Hz = self.fields_focus(
            f_input_field, n_bfp, focal_length, NA, xb, yb, zb,
            bead_center, bfp_sampling_n, num_orders, return_grid=False,
            total_field=False, inside_bead=False, H_field=True, verbose=verbose,
            grid=False
        )
    
        # Note: factor 1/2 incorporated later as 2 pi instead of 4 pi
        Px = (np.conj(Hz)*Ey - np.conj(Hy)*Ez).real
        Py = (np.conj(Hx)*Ez - np.conj(Hz)*Ex).real
        Pz = (np.conj(Hy)*Ex - np.conj(Hx)*Ey).real
        Psca = np.sum((Px * x + Py * y + Pz * z) * w)

        return Psca * (self.bead_diameter * 0.51)**2 * 2 * np.pi


def _R_phi(cos_phi, sin_phi):
    return np.asarray([[cos_phi, -sin_phi, 0],
                        [sin_phi, cos_phi, 0],
                        [0, 0, 1]])


def _R_th(cos_th, sin_th):
    return np.asarray([[cos_th, 0, sin_th],
                        [0,  1,  0],
                        [-sin_th, 0, cos_th]])

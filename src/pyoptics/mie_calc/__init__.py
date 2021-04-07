from .legendre import *

import numpy as np
import scipy.special as sp
from scipy.special import jv, yv


class MieCalc:
    def __init__(self, bead_diameter=1e-6, n_bead=1.5, n_medium=1.33,
              lambda_vac=1064e-9):
              self.bead_diameter = bead_diameter
              self.n_bead = n_bead
              self.n_medium = n_medium
              self.lambda_vac = lambda_vac
              self._k = 2 * np.pi * n_medium / lambda_vac
              self._k1 = 2 * np.pi * n_bead / lambda_vac

    def size_param(self):
        return np.pi * self.n_medium * self.bead_diameter / self.lambda_vac

    def nrel(self):
        return self.n_bead / self.n_medium

    def number_of_orders(self):
        size_param = self.size_param()
        return int(np.round(size_param + 4 * size_param**(1/3) + 2.0))

    def ab_coeffs(self, num_orders=None):
        nrel = self.nrel()
        size_param = self.size_param()
        y = nrel * size_param

        if num_orders is None:
            n_coeffs = self.number_of_orders()
        else:
            n_coeffs = int(np.max((np.abs(num_orders), 1)))

        # Downward recurrence for psi_n(x)'/psi_n(x) (logarithmic derivative)
        # See Bohren & Huffman, p. 127
        nmx = int(np.ceil(max(abs(y), n_coeffs)) + 15)
        D = np.zeros(nmx, dtype='complex128')
        for n in range(nmx - 2, -1, -1):
            rn = n + 2
            D[n] = rn / y - 1/(D[n + 1] + rn/y)

        psi_n, _ = sp.riccati_jn(n_coeffs, size_param)
        psi_n_1 = psi_n[0:n_coeffs]
        psi_n = psi_n[1:]

        ric_yn, _ = sp.riccati_yn(n_coeffs, size_param)
        ksi_n = psi_n + 1j*ric_yn[1:]
        ksi_n_1 = psi_n_1 + 1j*ric_yn[0:n_coeffs]

        Dn = D[0:n_coeffs]
        n = np.arange(1, n_coeffs + 1)
        an = ((Dn/nrel + n/size_param) * psi_n - psi_n_1) / ((Dn/nrel + n/size_param)
                * ksi_n - ksi_n_1)
        bn = ((nrel * Dn + n/size_param) * psi_n - psi_n_1) / (
            (nrel * Dn + n/size_param) * ksi_n - ksi_n_1)

        return an, bn

    def cd_coeffs(self, num_orders=None):
        nrel = self.nrel()
        size_param = self.size_param()
        y = nrel * size_param

        if num_orders is None:
            n_coeffs = self.number_of_orders()
        else:
            n_coeffs = int(np.max((np.abs(num_orders), 1)))

        n = np.arange(1, n_coeffs + 1)

        jnx = sp.spherical_jn(n, size_param)
        jn_1x = np.append(np.sin(size_param)/size_param, jnx[0:n_coeffs - 1])
        #jnx = jnx[1:]

        jny = sp.spherical_jn(n, y)
        jn_1y = np.append(np.sin(y)/y, jny[0:n_coeffs - 1])
        #jny = jny[1:]

        ynx = sp.spherical_yn(n, size_param)
        yn_1x = np.append(-np.cos(size_param)/size_param, ynx[0:n_coeffs - 1])
        hnx = jnx + 1j*ynx
        hn_1x = jn_1x + 1j*yn_1x


        cn = (jnx * (size_param * hn_1x - n * hnx) -
                    hnx * (jn_1x * size_param - n * jnx)) / (jny *
                    (hn_1x * size_param - n * hnx) - hnx * (jn_1y * y - n * jny))

        dn = (nrel * jnx * (size_param * hn_1x - n * hnx) - nrel * hnx *
            (size_param * jn_1x - n * jnx)) / (nrel**2 * jny * (
            size_param * hn_1x - n * hnx) - hnx * (y * jn_1y - n * jny))

        return cn, dn

    def fields_in_focus(self, n_BFP=1.0,
                        focal_length=4.43e-3, NA=1.2,
                        x=0, y=0, z=0, bead_center=(0,0,0),
                        bfp_sampling_n=31, num_orders=None,
                        return_grid=False,  total_field=True,
                        inside_bead=True, verbose=False):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        # TODO: warning criteria for undersampling/aliasing
        #M = int(np.max((31, 2 * NA**2 * np.max(np.abs(z)) /
        #        (np.sqrt(self.n_medium**2 - NA**2) * self.lambda_vac))))
        #if M > bfp_sampling_n:
        #    print('bfp_sampling_n lower than recommendation for convergence')

        self._init_local_coordinates(x,y,z, bead_center)
        self._get_mie_coefficients(num_orders)
        self._init_back_focal_plane(n_BFP, focal_length, NA, bfp_sampling_n)
        if verbose:
            print('Hankel functions')
        self._init_hankel()
        if verbose:
            print('Legendre functions')
        self._init_legendre(outside=True)


        Ex = np.zeros(self._XYZshape, dtype='complex128')
        Ey = np.zeros(self._XYZshape, dtype='complex128')
        Ez = np.zeros(self._XYZshape, dtype='complex128')
        if verbose:
            print('External field')
        self._calculate_field_speed_mem(Ex, Ey, Ez, False, total_field,
                                        bead_center)

        if inside_bead:
            if verbose:
                print('Bessel functions')
            self._init_bessel()
            if verbose:
                print('Legendre functions')
            self._init_legendre(outside=False)
            if verbose:
                print('Internal field')
            self._calculate_field_speed_mem(Ex, Ey, Ez, True, total_field,
                                            bead_center)

        ks = self._k * NA / self.n_medium
        dk = ks / (bfp_sampling_n - 1)
        phase = 1j * focal_length * (np.exp(-1j * self._k * focal_length) *
                dk**2 / (2 * np.pi))
        Ex *= phase
        Ey *= phase
        Ez *= phase

        Ex = np.squeeze(Ex)
        Ey = np.squeeze(Ey)
        Ez = np.squeeze(Ez)

        if return_grid:
            X, Y, Z = np.meshgrid(x, y, z)
            X = np.squeeze(X)
            Y = np.squeeze(Y)
            Z = np.squeeze(Z)
            return Ex, Ey, Ez, X, Y, Z
        else:
            return Ex, Ey, Ez

    def _init_local_coordinates(self, x=0, y=0, z=0, bead_center=(0,0,0)):
        # Set up coordinate system around bead
        X, Y, Z = np.meshgrid(x, y, z)
        self._XYZshape = X.shape

        # Local coordinate system around the bead
        Xlocal = X.reshape((1, -1)) - bead_center[0]
        Ylocal = Y.reshape((1, -1)) - bead_center[1]
        Zlocal = Z.reshape((1, -1)) - bead_center[2]

        self._R = np.hypot(np.hypot(Xlocal, Ylocal), Zlocal)
        self._outside = self._R > self.bead_diameter / 2
        self._inside = self._R <= self.bead_diameter / 2
        self._Xin = Xlocal[self._inside]
        self._Yin = Ylocal[self._inside]
        self._Zin = Zlocal[self._inside]
        self._Xout = Xlocal[self._outside]
        self._Yout = Ylocal[self._outside]
        self._Zout = Zlocal[self._outside]

    def _get_mie_coefficients(self, num_orders=None):
        # Get scattering and internal field coefficients
        an, bn = self.ab_coeffs(num_orders)
        cn, dn = self.cd_coeffs(num_orders)

        self._n_coeffs = an.shape[0]
        self._an = an
        self._bn = bn
        self._cn = cn
        self._dn = dn

    def _init_back_focal_plane(self, n_BFP, focal_length, NA, bfp_sampling_n):
        npupilsamples = (2*bfp_sampling_n - 1)

        x_BFP = np.linspace(-focal_length * NA / self.n_medium,
                            focal_length * NA / self.n_medium,
                            num=npupilsamples)

        X_BFP, Y_BFP = np.meshgrid(x_BFP, x_BFP)
        R_BFP = np.hypot(X_BFP, Y_BFP)

        # temporarily Gaussian
        w0 = 0.9 * focal_length * NA / self.n_medium
        Ein = np.exp(-(X_BFP**2 + Y_BFP**2) / w0**2)
        aperture = R_BFP > focal_length * NA / self.n_medium
        Ein[aperture] = 0
        self._Th = np.real(np.arcsin((R_BFP + 0j) / focal_length))
        self._Th[aperture] = 0
        self._Phi = np.arctan2(Y_BFP, X_BFP)
        self._Phi[R_BFP == 0] = 0
        self._Phi[aperture] = 0

        # Calculate properties of the plane waves
        self._Kz = self._k * np.cos(self._Th)
        self._Kp = self._k * np.sin(self._Th)
        self._Kx = self._Kp * np.cos(self._Phi)
        self._Ky = self._Kp * np.sin(self._Phi)

        # Transform the input wavefront to a spherical one, after refracting on
        # the Gaussian reference sphere [2], Ch. 3. The field magnitude changes
        # because of the different media, and because of the angle (preservation
        # of power in a beamlet). Finally, incorporate factor 1/kz of the integrand

        self._Einf = (np.sqrt(n_BFP / self.n_medium) * Ein *
                      np.sqrt(np.cos(self._Th)) / self._Kz)

        # Get p- and s-polarized parts
        self._Einf_theta = self._Einf * np.cos(self._Phi)
        self._Einf_phi = self._Einf * np.sin(self._Phi)

    def _init_hankel(self):
        # Precompute the spherical Hankel functions and derivatives that only depend
        # on the r coordinate. These functions will not change for any rotation of
        # the coordinate system.
        self._ro = self._R[self._outside]

        # Only calculate the spherical bessel function for unique values of k0r
        self._k0r  = self._k * self._ro
        k0r_unique, inverse = np.unique(self._k0r, return_inverse=True)
        sqrt_x = np.sqrt(0.5 * np.pi / k0r_unique)
        krh_1 = np.sin(self._k0r) - 1j*np.cos(self._k0r)
        self._sphHankel = np.empty((self._n_coeffs, self._ro.shape[0]),
                                    dtype='complex128')
        self._krh = np.empty(self._sphHankel.shape, dtype='complex128')
        self._dkrh_dkr = np.empty(self._sphHankel.shape, dtype='complex128')

        for L in range(1, self._n_coeffs + 1):
            self._sphHankel[L - 1,:] = (sqrt_x * (sp.jv(L + 0.5, k0r_unique) +
                                        1j * sp.yv(L + 0.5, k0r_unique)))[inverse]
            self._krh[L - 1,:] = self._k0r * self._sphHankel[L - 1,:]
            self._dkrh_dkr[L - 1,:] = krh_1 - L * self._sphHankel[L - 1,:]
            krh_1 = self._krh[L - 1, :]

    def _init_bessel(self):
        # Precompute the spherical Bessel functions and related that only depend on
        # the r coordinate, for the fields inside of the sphere.
        self._ri = self._R[self._inside]
        self._k1r = self._k1 * self._ri
        k1r_unique, inverse = np.unique(self._k1r, return_inverse=True)
        self._sphBessel = np.zeros((self._n_coeffs, self._ri.shape[0]),
                                    dtype='complex128')
        self._jn_over_k1r = np.zeros(self._sphBessel.shape, dtype='complex128')
        self._jn_1 = np.zeros(self._sphBessel.shape, dtype='complex128')
        jprev = np.empty(self._ri.shape[0], dtype='complex128')
        jprev[self._k1r>0] = (np.sin(self._k1r[self._k1r > 0]) /
                                self._k1r[self._k1r > 0])
        jprev[self._k1r == 0] = 1

        for L in range(1, self._n_coeffs + 1):
            self._sphBessel[L - 1, :] = sp.spherical_jn(L, k1r_unique)[inverse]
            self._jn_over_k1r[L - 1, self._k1r > 0] = self._sphBessel[L - 1,
                                    self._k1r > 0] / self._k1r[self._k1r > 0]
            self._jn_1[L - 1, :] = jprev
            jprev = self._sphBessel[L - 1, :]
        # The limit of the Spherical Bessel functions for jn(x)/x == 0, except
        # for n == 1. Then it is 1/3. See https://dlmf.nist.gov/10.52
        # For n > 1 taken care of by np.zeros(...)
        self._jn_over_k1r[0, self._k1r == 0] = 1/3

    def _init_legendre(self, outside=True):

        s = self._Einf.shape
        if outside:
            local_coords = np.vstack((self._Xout, self._Yout, self._Zout))
            cos_th_shape = (s[0], s[1], self._k0r.size)
            r = self._ro
        else:
            local_coords = np.vstack((self._Xin, self._Yin, self._Zin))
            cos_th_shape = (s[0], s[1], self._k1r.size)
            r = self._ri
        # Calculating the Legendre polynomials is computationally intense, so
        # Loop over all cos(theta), in order to find the unique values of
        # cos(theta)
        # TODO: Consider using parity to shorten calculations by a factor ~2
        cosTs = np.empty(cos_th_shape)

        for m in range(s[1]):
            for p in range(s[0]):
                if self._Einf[p, m] == 0:
                    continue
                # Rotate the coordinate system such that the x-polarization on the
                # bead coincides with theta polarization in global space
                # however, cos(theta) is the same for phi polarization!
                A = self._R_phi(self._Phi[p,m]) @ self._R_th(self._Th[p,m])
                coords = A.T @ local_coords
                Zvl_s = coords[2,:]

                # Retrieve an array of all values of cos(theta)
                if outside:
                    cosTs[p, m, :] = Zvl_s / r # cos(theta)
                else:
                    cosTs[p, m, r > 0] = Zvl_s[r > 0] / r[r > 0]
                    cosTs[p, m, r == 0] = 1


        cosTs = np.reshape(cosTs, cosTs.size)
        # rounding errors may make cos(theta) > 1 or < -1. Fix it to [-1..1]
        cosTs[cosTs > 1] = 1
        cosTs[cosTs < -1] = -1
        # Get the unique values of cos(theta) in the array
        cosT_unique, inverse = np.unique(cosTs, return_inverse=True)
        self._inverse = np.reshape(inverse, cos_th_shape)
        self._alp = np.zeros((self._n_coeffs, cosT_unique.size))
        self._alp_sin = np.zeros((self._n_coeffs, cosT_unique.size))
        self._alp_deriv = np.zeros((self._n_coeffs, cosT_unique.size))
        alp_prev = None # unique situation that for n == 1,
                        # the previous Assoc. Legendre Poly. isn't required

        for L in range(1, self._n_coeffs + 1):
            self._alp[L - 1,:] = associated_legendre(L, cosT_unique)
            self._alp_sin[L - 1, :] = associated_legendre_over_sin_theta(L,
                                    cosT_unique, self._alp[L - 1,:])
            self._alp_deriv[L - 1, :] = associated_legendre_dtheta(L, cosT_unique,
                                    (self._alp[L - 1,:], alp_prev))
            alp_prev = self._alp[L - 1,:]

    def _calculate_field_speed_mem(self, Ex, Ey, Ez, internal: bool,
                                   total_field: bool,
                                   bead_center: tuple):
        # Calculate the internal & external field from precalculated,
        # but compressed, Legendre polynomials
        # Compromise between speed and memory use
        E = np.empty((3, self._R.shape[1]), dtype='complex128')
        if internal:
            r = self._ri
            local_coords = np.vstack((self._Xin, self._Yin, self._Zin))
            region = np.squeeze(self._inside )
        else:
            r = self._ro
            local_coords = np.vstack((self._Xout, self._Yout, self._Zout))
            region = np.squeeze(self._outside)

        # preallocate memory for expanded legendre derivatives
        alp_expanded = np.empty((self._n_coeffs, r.size))
        alp_sin_expanded = np.empty((self._n_coeffs, r.size))
        alp_deriv_expanded = np.empty((self._n_coeffs, r.size))

        s = self._Einf.shape
        cosT = np.empty(r.shape)
        for m in range(s[1]):
            for p in range(s[0]):

                A = self._R_phi(self._Phi[p,m]) @ self._R_th(self._Th[p,m])
                coords = A.T @ local_coords
                Xvl_s = coords[0,:]
                Yvl_s = coords[1,:]
                Zvl_s = coords[2,:]

                if internal:
                    cosT[r > 0] = Zvl_s[r > 0] / r[r > 0]
                    cosT[r == 0] = 1
                else:
                    cosT = Zvl_s / r
                cosT[cosT > 1] = 1
                cosT[cosT < -1] = -1
                phil = np.arctan2(Yvl_s, Xvl_s)

                E[:] = 0

                # Expand the legendre derivatives from the unique version of
                # cos(theta)
                alp_expanded[:] = self._alp[:, self._inverse[p,m]]
                alp_sin_expanded = self._alp_sin[:, self._inverse[p,m]]
                alp_deriv_expanded = self._alp_deriv[:, self._inverse[p,m]]

                if internal:
                    E[:, np.squeeze(self._inside)] = self._internal_field_fixed_r(
                                    alp_expanded,
                                    alp_sin_expanded, alp_deriv_expanded,
                                    cosT, phil)
                else:
                    E[:, np.squeeze(self._outside)] = self._scattered_field_fixed_r(
                                alp_expanded, alp_sin_expanded,
                                alp_deriv_expanded, cosT, phil, total_field)

                E = np.matmul(A, E)

                E[:, region] *= self._Einf_theta[p,m] * np.exp(1j *
                    (self._Kx[p,m] * bead_center[0] +
                     self._Ky[p,m] * bead_center[1] +
                     self._Kz[p,m] * bead_center[2]))

                Ex[:,:,:] += np.reshape(E[0,:], self._XYZshape)
                Ey[:,:,:] += np.reshape(E[1,:], self._XYZshape)
                Ez[:,:,:] += np.reshape(E[2,:], self._XYZshape)

                # phi-polarization
                A = (self._R_phi(self._Phi[p,m]) @ self._R_th(self._Th[p,m]) @
                        self._R_phi(-np.pi/2))

                coords = A.T @ local_coords
                Xvl_s = coords[0,:]
                Yvl_s = coords[1,:]
                Zvl_s = coords[2,:]

                # cosT is still valid from the previous time, but phil is not as
                # X and Y got swapped and a sign got flipped
                phil = np.arctan2(Yvl_s, Xvl_s)

                if internal:
                    E[:, region] = self._internal_field_fixed_r(
                                    alp_expanded,
                                    alp_sin_expanded, alp_deriv_expanded,
                                    cosT, phil)
                else:
                    E[:, region] = self._scattered_field_fixed_r(
                                alp_expanded, alp_sin_expanded,
                                alp_deriv_expanded, cosT, phil, total_field)

                E = np.matmul(A, E)

                E[:, region] *= self._Einf_phi[p,m] * np.exp(1j *
                    (self._Kx[p,m] * bead_center[0] +
                     self._Ky[p,m] * bead_center[1] +
                     self._Kz[p,m] * bead_center[2]))

                Ex[:,:,:] += np.reshape(E[0,:], self._XYZshape)
                Ey[:,:,:] += np.reshape(E[1,:], self._XYZshape)
                Ez[:,:,:] += np.reshape(E[2,:], self._XYZshape)


    def _scattered_field_fixed_r(self, alp: np.ndarray, alp_sin: np.ndarray,
                                alp_deriv: np.ndarray,
                                cos_theta: np.ndarray, phi: np.ndarray,
                                total_field=True):

        # Radial, theta and phi-oriented fields
        Er = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
        Et = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
        Ep = np.zeros((1,cos_theta.shape[0]), dtype='complex128')

        sinT = np.sqrt(1 - cos_theta**2)
        cosP = np.cos(phi)
        sinP = np.sin(phi)

        an = self._an
        bn = self._bn
        krh = self._krh
        dkrh_dkr = self._dkrh_dkr
        k0r = self._k0r
        L = np.arange(start=1, stop=self._n_coeffs + 1)
        C1 = 1j**(L + 1) * (2 * L + 1)
        C2 = C1 / (L * (L + 1))
        for L in range(1, self._n_coeffs + 1):
            Er += C1[L-1] * an[L - 1] * krh[L - 1,:] * alp[L-1,:]

            Et += C2[L-1] * (an[L - 1] *
                    dkrh_dkr[L - 1,:] * alp_deriv[L - 1,:] + 1j*bn[L - 1] *
                    krh[L - 1,:] * alp_sin[L - 1,:])

            Ep += C2[L-1] * (an[L - 1] *
                dkrh_dkr[L - 1,:] * alp_sin[L - 1,:] + 1j * bn[L - 1] *
                krh[L - 1,:] * alp_deriv[L - 1,:])

        Er *= -np.cos(phi) / (k0r)**2
        Et *= -np.cos(phi) / (k0r)
        Ep *= np.sin(phi) / (k0r)
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

    def _internal_field_fixed_r(self, alp: np.ndarray, alp_sin: np.ndarray,
                          alp_deriv: np.ndarray,
                          cos_theta: np.ndarray, phi: np.ndarray):

        # Radial, theta and phi-oriented fields
        Er = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
        Et = np.zeros((1,cos_theta.shape[0]), dtype='complex128')
        Ep = np.zeros((1,cos_theta.shape[0]), dtype='complex128')

        sinT = np.sqrt(1 - cos_theta**2)
        cosP = np.cos(phi)
        sinP = np.sin(phi)

        #short hand for readability:
        cn = self._cn
        dn = self._dn
        sphBessel = self._sphBessel
        jn_over_k1r = self._jn_over_k1r
        jn_1 = self._jn_1

        for n in range(1, self._n_coeffs + 1):
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


        Er *= -np.cos(phi)
        Et *= -np.cos(phi)
        Ep *= -np.sin(phi)
        # Cartesian components
        Ex = Er * sinT * cosP + Et * cos_theta * cosP - Ep * sinP
        Ey = Er * sinT * sinP + Et * cos_theta * sinP + Ep * cosP
        Ez = Er * cos_theta - Et * sinT
        return np.concatenate((Ex, Ey, Ez), axis=0)

    def _R_phi(self, phi):
        return np.asarray([[np.cos(phi), -np.sin(phi), 0],
                           [np.sin(phi), np.cos(phi), 0],
                           [0, 0, 1]])

    def _R_th(self, th):
        return np.asarray([[np.cos(th), 0, np.sin(th)],
                           [0,  1,  0],
                           [-np.sin(th), 0, np.cos(th)]])

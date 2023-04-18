import numpy as np
import scipy.special as sp

class Bead:
    def __init__(self, bead_diameter=1e-6, n_bead=1.57, n_medium=1.33,
              lambda_vac=1064e-9):
        """Create a new instance of the MieCalc class. The class describes 
        the situation of a spherical bead with a diameter of bead_diameter 
        (in meters) and having a refractive index n_bead, where the bead is 
        embedded in a medium with a refractive index n_medium. The wavelength 
        for any calculation is lambda_vac, which is the wavelength in vacuum 
        (in meters)
        
        Parameters
        ----------
        bead_diameter : Diameter of the bead in meters
        n_bead : refractive index of the bead
        n_medium : refractive index of the medium
        lambda_vac : wavelength of the light in meters, in vacuum (so not the
        medium)
        """

        self.bead_diameter = bead_diameter
        self.n_bead = n_bead
        self.n_medium = n_medium
        self.lambda_vac = lambda_vac
    
    def __repr__(self) -> str:
        return (
            f'Diameter: {self.bead_diameter} [m]\n'
            f'Refractive index: {self.n_bead}\n'
            f'Refractive index medium: {self.n_medium}\n'
            f'Wavelength: {self.lambda_vac} [m]\n'
        )
    
    @property
    def k(self) -> float:
        return 2 * np.pi * self.n_medium / self.lambda_vac
    
    @property
    def k1(self) -> float:
        return 2 * np.pi * self.n_bead / self.lambda_vac

    @property
    def size_param(self):
        """Return the size parameter of the bead k*a, where k is the wave number
        in the medium and a is the radius of the bead
        
        Returns
        -------
        size parameter [-]
        """
        return np.pi * self.n_medium * self.bead_diameter / self.lambda_vac

    @property
    def nrel(self):
        """Return the relative refractive index n_bead/n_medium"""
        return self.n_bead / self.n_medium

    @property
    def number_of_orders(self):
        """Return the number of orders required to 'properly' approximate the 
        fields scattered by the bead. The criterion is the closest integer to 
        x + 4x^(1/3) + 2, where x is the size parameter of the bead [1] 
        
        1.  "Absorption and Scattering of Light by Small Particles", 
            Craig F. Bohren & Donald R. Huffman, p. 477
        
        Returns
        -------
        n : number of orders
        """

        size_param = self.size_param
        return int(np.round(size_param + 4 * size_param**(1/3) + 2.0))

    def ab_coeffs(self, num_orders=None):
        """Return the scattering coefficients for plane wave excitation of the
        bead. 
        
        Parameters
        ----------
        num_orders : determines the number of orders returned. If num_orders is
            None (default), the number of orders returned is determined by the
            method number_of_orders()
        
        Returns
        -------
        an : scattering coefficients
        bn : scattering coefficients
        """

        nrel = self.nrel
        size_param = self.size_param
        y = nrel * size_param

        if num_orders is None:
            n_coeffs = self.number_of_orders
        else:
            n_coeffs = int(np.max((np.abs(num_orders), 1)))

        # Downward recurrence for psi_n(x)'/psi_n(x) (logarithmic derivative)
        # See Bohren & Huffman, p. 127
        nmx = int(np.ceil(max(abs(y), n_coeffs)) + 15)
        D = np.zeros(nmx, dtype='complex128')
        for n in range(nmx - 2, -1, -1):
            rn = n + 2
            D[n] = rn / y - 1 / (D[n + 1] + rn / y)

        psi_n, _ = sp.riccati_jn(n_coeffs, size_param)
        psi_n_1 = psi_n[0:n_coeffs]
        psi_n = psi_n[1:]

        ric_yn, _ = sp.riccati_yn(n_coeffs, size_param)
        ksi_n = psi_n + 1j * ric_yn[1:]
        ksi_n_1 = psi_n_1 + 1j * ric_yn[0:n_coeffs]

        Dn = D[0:n_coeffs]
        n = np.arange(1, n_coeffs + 1)
        an = ((Dn / nrel + n / size_param) * psi_n - psi_n_1) / ((Dn / nrel + n / size_param)
                * ksi_n - ksi_n_1)
        bn = ((nrel * Dn + n / size_param) * psi_n - psi_n_1) / (
            (nrel * Dn + n / size_param) * ksi_n - ksi_n_1)

        return an, bn

    def cd_coeffs(self, num_orders=None):
        """Return the coefficients for the internal field of the bead.
        
        Parameters
        ----------
        num_orders : determines the number of orders returned. If num_orders is
            None (default), the number of orders returned is determined by the
            method number_of_orders()
        
        Returns
        -------
        cn : internal field coefficients
        dn : internal field coefficients
        """
        nrel = self.nrel
        size_param = self.size_param
        y = nrel * size_param

        if num_orders is None:
            n_coeffs = self.number_of_orders
        else:
            n_coeffs = int(np.max((np.abs(num_orders), 1)))

        n = np.arange(1, n_coeffs + 1)

        jnx = sp.spherical_jn(n, size_param)
        jn_1x = np.append(np.sin(size_param) / size_param, jnx[0:n_coeffs - 1])

        jny = sp.spherical_jn(n, y)
        jn_1y = np.append(np.sin(y) / y, jny[0:n_coeffs - 1])

        ynx = sp.spherical_yn(n, size_param)
        yn_1x = np.append(-np.cos(size_param) / size_param, ynx[0:n_coeffs - 1])
        hnx = jnx + 1j * ynx
        hn_1x = jn_1x + 1j * yn_1x


        cn = (jnx * (size_param * hn_1x - n * hnx) -
                    hnx * (jn_1x * size_param - n * jnx)) / (jny *
                    (hn_1x * size_param - n * hnx) - hnx * (jn_1y * y - n * jny))

        dn = (nrel * jnx * (size_param * hn_1x - n * hnx) - nrel * hnx *
            (size_param * jn_1x - n * jnx)) / (nrel**2 * jny * (
            size_param * hn_1x - n * hnx) - hnx * (y * jn_1y - n * jny))

        return cn, dn

    
    def extinction_eff(self, num_orders=None):
        """Return the extinction efficiency Qext (for plane wave excitation),
        defined as Qext = Cext/(pi r**2), where Cext is the exctinction cross
        section and r is the bead radius.

        Parameters
        ----------
        num_orders : determines the number of orders returned. If num_orders is
            None (default), the number of orders returned is determined by the
            method number_of_orders()
        
        Returns
        -------
        Qext : extinction efficiency
        """

        an, bn = self.ab_coeffs(num_orders=num_orders)
        C = 2 * (1 + np.arange(an.size)) + 1
        return (
            2 * self.size_param**-2 * 
            np.sum(C * (an + bn).real)
        )


    def scattering_eff(self, num_orders=None):
        """Return the scattering efficiency Qsca (for plane wave excitation),
        defined as Qsca = Csca/(pi r**2), where Csca is the scattering cross
        section and r is the bead radius.

        Parameters
        ----------
        num_orders : determines the number of orders returned. If num_orders is
            None (default), the number of orders returned is determined by the
            method number_of_orders()
        
        Returns
        -------
        Qsca : scattering efficiency
        """

        an, bn = self.ab_coeffs(num_orders=num_orders)
        C = 2 * (1 + np.arange(an.size)) + 1
        return (
            2 * self.size_param**-2 * 
            np.sum(C * (np.abs(an)**2 + np.abs(bn)**2))
        )

    
    def pressure_eff(self, num_orders=None):
        """Return the pressure efficiency Qpr (for plane wave excitation),
        defined as Qpr = Qext - Qsca <cos(theta)>, where <cos(theta)> is the
        mean scattering angle.

        Parameters
        ----------
        num_orders : determines the number of orders returned. If num_orders is
            None (default), the number of orders returned is determined by the
            method number_of_orders()
        
        Returns
        -------
        Qpr : pressure efficiency

        """

        an, bn = self.ab_coeffs(num_orders=num_orders)
        n = 1 + np.arange(an.size)
        C = 2 * n + 1
        C1 = n * (n + 2)/(n + 1)
        C2 = C/(n * (n + 1))
        an_1 = np.zeros(an.shape, dtype='complex128')
        bn_1 = np.zeros(an.shape, dtype='complex128')
        an_1[0:-2] = an[1:-1]
        bn_1[0:-2] = bn[1:-1]

        return self.extinction_eff(num_orders) - (
            4 * self.size_param**-2 * 
            np.sum(C1 * (an * np.conj(an_1) + 
            bn * np.conj(bn_1)).real + 
            C2 * (an * np.conj(bn)).real)
        )


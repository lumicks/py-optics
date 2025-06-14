import math

import numpy as np
import scipy.special as sp
from deprecated import deprecated

from lumicks.pyoptics.mathutils.integration.sphere import get_nearest_order


def _determine_integration_order(method: str, n_orders: int):
    """Helper function to generate reasonable defaults for the integration order
    for different integration methods, based on the number of Mie scattering orders. The presumption
    is that some kind of entity that is based on the squared field strength is being integrated.

    Parameters
    ----------
    method : str
        Integration method, one of "lebedev-laikov", "gauss-legendre" or
        "clenshaw-curtis".
    n_orders : int
        Number of Mie scattering orders in the Mie solution

    Returns
    -------
    int
        Integration order based on the integration method and number of Mie
        scattering orders. Must be > 0

    Raises
    ------
    ValueError
        Raised if an invalid integration method is passed
        Raised if `n_orders <= 0`
    """

    if (n := math.ceil(n_orders)) <= 0:
        raise ValueError(f"Invalid value for n_orders. Must be > 0, got {n_orders}")

    # Determine reasonable defaults for integrating over a certain bead size
    # Guesstimate based on P^1_n ** 2 ~ (1 - x ** 2) * (x ** (n - 1)) ** 2 ~ x ** 2n
    if method == "gauss-legendre":  # integration order m is accurate to x ** (2 * m - 1)
        integration_order = n + 1
    elif method == "clenshaw-curtis":  # integration order m is accurate to x ** m
        integration_order = 2 * n
    # lebedev-laikov
    elif method == "lebedev-laikov":
        # Get an integration order that is one level higher than the one matching 2 * n_orders
        # if no integration order is specified
        integration_order = get_nearest_order(2 * n, "lebedev-laikov")
    else:
        raise ValueError(f"Wrong type of integration method specified: {method}")

    return integration_order


class Bead:
    def __init__(
        self,
        bead_diameter: float = 1e-6,
        n_bead: float | complex = 1.57,
        n_medium: float = 1.33,
        lambda_vac: float = 1064e-9,
    ):
        """
        The `Bead` class describes the situation of a spherical bead with a diameter of
        `bead_diameter` (in meters) and having a refractive index `n_bead`, where the bead is
        embedded in a medium with a refractive index `n_medium`. The wavelength for any calculation
        is `lambda_vac`, which is the wavelength in vacuum (in meters)

        Parameters
        ----------
        bead_diameter : float, optional
            Diameter of the bead in meters, by default 1e-6
        n_bead : float, optional
            Refractive index of the bead, by default 1.57
        n_medium : float, optional
            Refractive index of the medium, by default 1.33
        lambda_vac : float, optional
            Wavelength of the light in meters, in vacuum (so not the medium), by default 1064e-9
        """

        self.bead_diameter = bead_diameter
        self.n_bead = n_bead
        self.n_medium = n_medium
        self.lambda_vac = lambda_vac

    def __str__(self) -> str:
        return (
            "Bead\n"
            "----\n"
            f"Diameter: {self.bead_diameter} [m]\n"
            f"Refractive index: {self.n_bead}\n"
            f"Refractive index medium: {self.n_medium}\n"
            f"Wavelength: {self.lambda_vac} [m]\n"
        )

    def __repr__(self) -> str:
        return (
            f"Bead(bead_diameter={self.bead_diameter}, "
            f"n_bead={self.n_bead}, "
            f"n_medium={self.n_medium}, "
            f"lambda_vac={self.lambda_vac})"
        )

    @property
    def k(self) -> float:
        """Returns the wave number `k` of the medium surrounding the bead.

        Returns
        -------
        float
            `k`, the wave number of the medium.
        """
        return 2 * np.pi * self.n_medium / self.lambda_vac

    @property
    def k1(self) -> float:
        """Returns the wave number `k1` of the material of the bead.

        Returns
        -------
        float
            `k1`, the wave number of the material of the bead
        """
        return 2 * np.pi * self.n_bead / self.lambda_vac

    @property
    def size_param(self) -> float:
        """
        Return the size parameter of the bead :math:`k a`, where :math:`k = 2 \\pi
        n_{medium}/\\lambda_{vac}`, the wave number in the medium and :math:`a` is the radius of the
        bead.

        Returns
        -------
        float
            The size parameter of the bead. [-]
        """
        return np.pi * self.n_medium * self.bead_diameter / self.lambda_vac

    @property
    def nrel(self) -> float | complex:
        """
        Return the relative refractive index n_bead/n_medium
        """
        return self.n_bead / self.n_medium

    @property
    @deprecated(reason="Renamed to `number_of_modes", version="0.7.0")
    def number_of_orders(self) -> int:
        return self.number_of_modes

    @property
    def number_of_modes(self) -> int:
        """
        Return the number of modes required to 'properly' approximate the fields scattered by the
        bead. The criterion is the closest integer to :math:`x + 4 x^{1/3} + 2`, where `x` is the
        size parameter of the bead [1]_. See also :py:meth:`size_param`.

        Returns
        -------
        int
            number of orders

        ..  [1] "Absorption and Scattering of Light by Small Particles", Craig F. Bohren & Donald R.
            Huffman, p. 477
        """

        size_param = self.size_param
        return int(np.round(size_param + 4 * size_param ** (1 / 3) + 2.0))

    def ab_coeffs(self, num_modes=None):
        """Return the scattering coefficients for plane wave excitation of the bead.

        Parameters
        ----------
        num_modes : int, optional
            Determines the number of modes returned. If `num_modes` is `None` (default), the
            number of orders returned is determined by the property `number_of_modes`

        Returns
        -------
        an : np.ndarray
            Scattering coefficients :math:`a_n`
        bn : np.ndarray
            scattering coefficients :math:`b_n`
        """

        nrel = self.nrel
        size_param = self.size_param
        y = nrel * size_param

        if num_modes is None:
            n_coeffs = self.number_of_modes
        else:
            n_coeffs = int(np.max((np.abs(num_modes), 1)))

        # Downward recurrence for psi_n(x)'/psi_n(x) (logarithmic derivative)
        # See Bohren & Huffman, p. 127
        nmx = int(np.ceil(max(abs(y), n_coeffs)) + 15)
        D = np.zeros(nmx, dtype="complex128")
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
        an = ((Dn / nrel + n / size_param) * psi_n - psi_n_1) / (
            (Dn / nrel + n / size_param) * ksi_n - ksi_n_1
        )
        bn = ((nrel * Dn + n / size_param) * psi_n - psi_n_1) / (
            (nrel * Dn + n / size_param) * ksi_n - ksi_n_1
        )

        return an, bn

    def cd_coeffs(self, num_modes=None):
        """
        Return the coefficients for the internal field of the bead.

        Parameters
        ----------
        num_modes : int, optional
            Determines the number of orders returned. If num_modes is `None` (default), the number
            of orders returned is determined by the property `number_of_modes`

        Returns
        -------
        cn : np.ndarray
            Internal field coefficients :math:`c_n`
        dn : np.ndarray
            Internal field coefficients :math:`d_n`
        """
        nrel = self.nrel
        size_param = self.size_param
        y = nrel * size_param

        if num_modes is None:
            n_coeffs = self.number_of_modes
        else:
            n_coeffs = int(np.max((np.abs(num_modes), 1)))

        n = np.arange(1, n_coeffs + 1)

        jnx = sp.spherical_jn(n, size_param)
        jn_1x = np.append(np.sin(size_param) / size_param, jnx[0 : n_coeffs - 1])

        jny = sp.spherical_jn(n, y)
        jn_1y = np.append(np.sin(y) / y, jny[0 : n_coeffs - 1])

        ynx = sp.spherical_yn(n, size_param)
        yn_1x = np.append(-np.cos(size_param) / size_param, ynx[0 : n_coeffs - 1])
        hnx = jnx + 1j * ynx
        hn_1x = jn_1x + 1j * yn_1x

        cn = (jnx * (size_param * hn_1x - n * hnx) - hnx * (jn_1x * size_param - n * jnx)) / (
            jny * (hn_1x * size_param - n * hnx) - hnx * (jn_1y * y - n * jny)
        )

        dn = (
            nrel * jnx * (size_param * hn_1x - n * hnx)
            - nrel * hnx * (size_param * jn_1x - n * jnx)
        ) / (nrel**2 * jny * (size_param * hn_1x - n * hnx) - hnx * (y * jn_1y - n * jny))

        return cn, dn

    def extinction_eff(self, num_modes=None):
        """Return the extinction efficiency `Qext` (for plane wave excitation), defined as
        :math:`Q_{ext} = C_{ext}/(\\pi r^2)`, where :math:`C_{ext}` is the exctinction cross section
        and :math:`r` is the bead radius.

        Parameters
        ----------
        num_modes : int, optional,
            Determines the number of orders returned. If num_modes is `None` (default), the number
            of orders returned is determined by the property `number_of_modes`

        Returns
        -------
        float
            Extinction efficiency
        """

        an, bn = self.ab_coeffs(num_modes=num_modes)
        C = 2 * (1 + np.arange(an.size)) + 1
        return 2 * self.size_param**-2 * np.sum(C * (an + bn).real)

    def scattering_eff(self, num_modes=None):
        """Return the scattering efficiency `Qsca` (for plane wave excitation),
        defined as :math:`Q_{sca} = C_{sca}/(\\pi r^2)`, where :math:`C_{sca}` is the scattering
        cross section and :math:`r` is the bead radius.

        Parameters
        ----------
        num_modes : int, optional
            Determines the number of orders returned. If num_modes is `None` (default), the number
            of orders returned is determined by the property `number_of_modes`

        Returns
        -------
        float
            Scattering efficiency
        """

        an, bn = self.ab_coeffs(num_modes=num_modes)
        C = 2 * (1 + np.arange(an.size)) + 1
        return 2 * self.size_param**-2 * np.sum(C * (np.abs(an) ** 2 + np.abs(bn) ** 2))

    def pressure_eff(self, num_modes=None):
        """
        Return the pressure efficiency `Qpr` (for plane wave excitation), defined as :math:`Q_{pr} =
        Q_{ext} - Q_{sca} <\\cos(\\theta)>`, where :math:`<\\cos(\\theta)>` is the mean scattering
        angle.

        Parameters
        ----------
        num_modes : int, optional
            Determines the number of orders returned. If num_modes is
            `None` (default), the number of orders returned is determined by the property
            `number_of_modes`

        Returns
        -------
        float
            Pressure efficiency

        """

        an, bn = self.ab_coeffs(num_modes=num_modes)
        n = 1 + np.arange(an.size)
        C = 2 * n + 1
        C1 = n * (n + 2) / (n + 1)
        C2 = C / (n * (n + 1))
        an_1 = np.zeros(an.shape, dtype="complex128")
        bn_1 = np.zeros(an.shape, dtype="complex128")
        an_1[0:-2] = an[1:-1]
        bn_1[0:-2] = bn[1:-1]

        return self.extinction_eff(num_modes) - (
            4
            * self.size_param**-2
            * np.sum(
                C1 * (an * np.conj(an_1) + bn * np.conj(bn_1)).real + C2 * (an * np.conj(bn)).real
            )
        )

import math
from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

from .lebedev_laikov import get_integration_locations as ll_get_integration_locations
from .lebedev_laikov import get_nearest_order


def determine_integration_order(method: str, n_orders: int):
    """Helper function to generate reasonable defaults for the integration order
    for different integration methods, based on the number of Mie scattering
    orders. The presumption is that some kind of entity that is based on the
    squared field strength is being integrated.

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
    if method not in ["lebedev-laikov", "gauss-legendre", "clenshaw-curtis"]:
        raise ValueError(f"Wrong type of integration method specified: {method}")

    if (n := math.ceil(n_orders)) <= 0:
        raise ValueError(f"Invalid value for n_orders. Must be > 0, got {n_orders}")

    # Determine reasonable defaults for integrating over a certain bead size
    # Guesstimate based on P^1_n ** 2 ~ (1 - x ** 2) * (x ** (n - 1)) ** 2 ~ x ** 2n
    if method == "gauss-legendre":  # integration order m is accurate to x ** (2 * m - 1)
        integration_order = n + 1
    elif method == "clenshaw-curtis":  # integration order m is accurate to x ** m
        integration_order = 2 * n
    # lebedev-laikov
    else:
        # Get an integration order that is one level higher than the one matching 2 * n_orders
        # if no integration order is specified
        integration_order = get_nearest_order(2 * n)
    return integration_order


@lru_cache(maxsize=8)
def get_integration_locations(integration_order: int, method: str):
    if method == "lebedev-laikov":
        return [np.asarray(c) for c in ll_get_integration_locations(integration_order)]
    if method == "gauss-legendre":
        z, w = roots_legendre(integration_order)
    elif method == "clenshaw-curtis":
        z, w = clenshaw_curtis_weights(integration_order)
    else:
        raise RuntimeError(f"Unsupported integration method {method}")

    phi = np.arange(1, 2 * integration_order + 1) * np.pi / integration_order
    x, y = np.cos(phi), np.sin(phi)
    sin_theta = ((1 - z) * (1 + z)) ** 0.5
    x = (sin_theta[:, np.newaxis] * x[np.newaxis, :]).flatten()
    y = (sin_theta[:, np.newaxis] * y[np.newaxis, :]).flatten()
    z = np.repeat(z, phi.size)
    w = np.repeat(w, phi.size) * 0.25 / integration_order
    return x, y, z, w


@lru_cache(maxsize=8)
def clenshaw_curtis_weights(integration_order: int):
    """Generate sample locations and weigths for Clenshaw-Curtis integration.

    Based on the paper by J. Waldvogel (2006) DOI: 10.1007/s10543-006-0045-4

    Parameters
    ----------
    integration_order : int
        Order of the integration, must be ≥ 2.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple (x, w): x contains the locations where the function to be integrated is sampled, and
        w contains the weights for each sample.

    Raises
    ------
    ValueError
        Raised if integration_order is < 2.
    """
    if integration_order < 2:
        raise ValueError("Only integration orders of 2 and higher are supported")
    integration_order = math.floor(integration_order)
    w0cc = (integration_order**2 - 1 + (integration_order & 1)) ** -1
    gk = np.full(integration_order, fill_value=-w0cc)
    gk[integration_order // 2] *= -((2 - integration_order % 2) * integration_order - 1)

    vk = np.empty(integration_order)
    vk[: integration_order // 2] = 2 * (1 - 4 * np.arange(integration_order // 2) ** 2) ** -1.0
    vk[integration_order // 2] = (integration_order - 3) * (
        2 * (integration_order // 2) - 1
    ) ** -1.0 - 1
    if integration_order & 1:
        gk[integration_order // 2 + 1] = gk[integration_order // 2]
        vk[integration_order - 1 : integration_order // 2 : -1] = vk[1 : integration_order // 2 + 1]
    else:
        vk[integration_order - 1 : integration_order // 2 : -1] = vk[1 : integration_order // 2]

    w = np.empty(integration_order + 1, dtype="complex128")
    np.fft.ifft(gk + vk, out=w[:integration_order])
    w[-1] = w[0]
    return np.cos(np.arange(integration_order + 1) * np.pi / integration_order), w.real


def annulus_rule(n_r: int, n_t: int | None = None, r_inner: float = 0.0, r_outer: float = 1.0):
    """An integration rule for circular domains, based on _[1].

    Parameters
    ----------
    n_r : int
        Number of points / order in the radial direction
    n_t : int | None, optional
        Number of points in the angular direction, by default None. If None, the number of points is
        `4 * n_r + 3`.
    r_inner : float, optional
        Inner radius of the circular domain, by default 0.0
    r_outer : float, optional
        Outer radius of the circular domain, by default 1.0

    Returns
    -------
    tuple[np.array, np.array, np.array]
        Tuple of Numpy arrays that contain the locations in x and y, and the weight factors for each
        location, respectively. The weight factor sums up to the area of the domain of integration.

    .. [1] William H. Peirce, "Numerical Integration Over the Planar Annulus,",  Journal of the
        Society for Industrial and Applied Mathematics, Vol. 5, No. 2 (Jun., 1957), pp. 66-73
    """
    k = 4 * n_r + 3 if n_t is None else n_t
    theta_i = 2 * np.pi / (k + 1) * np.arange(1, k + 2)

    xi, wi = roots_legendre(n_r + 1)

    # Adjust range
    ri = ((r_inner**2 + r_outer**2) / 2.0 + (r_outer**2 - r_inner**2) / 2.0 * xi) ** 0.5
    wi = 0.5 * wi / (k + 1) * np.pi * (r_outer + r_inner) * (r_outer - r_inner)

    x = np.zeros(xi.size * theta_i.size)
    y = np.zeros_like(x)
    w = np.tile(wi, k + 1)
    ri, theta_i = np.meshgrid(ri, theta_i)
    x, y = (ri * np.cos(theta_i)).flatten(), (ri * np.sin(theta_i)).flatten()

    return x, y, w

import math
from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

from .lebedev_laikov import get_integration_locations as ll_get_integration_locations
from .lebedev_laikov import get_nearest_order as ll_get_nearest_order


def get_nearest_order(order: int, method: str):
    if method != "lebedev-laikov":
        return max(2, math.floor(order))
    else:
        return ll_get_nearest_order(order)


@lru_cache(maxsize=8)
def get_integration_locations(integration_order: int, method: str):
    if method == "lebedev-laikov":
        return [np.asarray(c) for c in ll_get_integration_locations(integration_order)]
    if method == "gauss-legendre":
        z, w = roots_legendre(integration_order)
    elif method == "clenshaw-curtis":
        z, w = _clenshaw_curtis_weights(integration_order)
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
def _clenshaw_curtis_weights(integration_order: int):
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

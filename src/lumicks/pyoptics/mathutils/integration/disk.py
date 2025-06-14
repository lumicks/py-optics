import numpy as np
from scipy.special import roots_legendre

from .takaki import get_disk_rule as tk_get_disk_rule


def get_integration_locations(integration_order: int | tuple[int, int], method: str):
    if method == "peirce":
        order = (
            integration_order if isinstance(integration_order, tuple) else (integration_order, None)
        )
        return annulus_rule(order[0], order[1], r_inner=0, r_outer=1.0)
    if method == "takaki":
        return tk_get_disk_rule(integration_order)
    if method == "lether":
        return disk_rule_lether(integration_order)


def annulus_rule(n_r: int, n_t: int | None = None, r_inner: float = 0.0, r_outer: float = 1.0):
    """An integration rule for circular domains, based on _[1].

    Parameters
    ----------
    n_r : int
        Number of points / order in the radial direction
    n_t : int | None, optional
        Number of points in the angular direction, by default None. If None, the number of points is
        `4 * (n_r + 1)`.
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
    xi, wi = roots_legendre(n_r)

    # Adjust range
    ri = ((r_inner**2 + r_outer**2) / 2.0 + (r_outer**2 - r_inner**2) / 2.0 * xi) ** 0.5
    wi = 0.5 * wi / (k + 1) * np.pi * (r_outer + r_inner) * (r_outer - r_inner)

    x = np.zeros(xi.size * theta_i.size)
    y = np.zeros_like(x)
    w = np.tile(wi, k + 1)
    ri, theta_i = np.meshgrid(ri, theta_i)
    x, y = (ri * np.cos(theta_i)).flatten(), (ri * np.sin(theta_i)).flatten()

    return x, y, w


def disk_rule_lether(order: int):
    """An integration rule for the unit disk, based on _[2].

    Parameters
    ----------
    order : int
        Order of the integration rule, exact to degree 2n - 1, where n is the order.

    Returns
    -------
    tuple[np.array, np.array, np.array]
        Tuple of Numpy arrays that contain the locations in x and y, and the weight factors for each
        location, respectively. The weight factors sum up to π.

    .. [2] Frank G. Lether, "A Generalized Product Rule For The Circle,"  SIAM Journal on Numerical
        Analysis, Volume 8, Issue 2, Pages 249 - 253

    """
    bv, Bv = roots_legendre(order)
    mu = np.arange(1, order + 1)[:, np.newaxis]
    wd = np.pi / (order + 1) * Bv[np.newaxis, :] * np.sin(mu * np.pi / (order + 1)) ** 2
    xmu = np.tile(np.cos(mu * np.pi / (order + 1)), (1, order))
    ymuv = bv * np.sin(mu * np.pi / (order + 1))
    return xmu.reshape(1, -1).squeeze(), ymuv.reshape(1, -1).squeeze(), wd.reshape(1, -1).squeeze()

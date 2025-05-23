import math

from scipy.integrate import lebedev_rule

_ORDER = [3 + n * 2 for n in range(15)] + [35 + n * 6 for n in range(17)]


def get_nearest_order(order: int):
    """For a given `order`, return the nearest (higher) order that we have sampling coordinates and
    weight factors for.

    Parameters
    ----------
    order : int
        Order to check against. Must be >= 1.

    Returns
    -------
    nearest_order : int
        The nearest order with known coordinates and coefficients.

    Raises
    ------
    ValueError
        Orders smaller than 1 and larger than 131 are not supported and have an exception as a
        result.
    """
    int_order = int(order)
    if int_order > 131 or order < 1:
        raise ValueError(f"A value of {order} is not supported")

    return min(filter(lambda x: x >= int_order, _ORDER))


def get_integration_locations(order: int):
    """Retrieve the integration locations and weight factors for a certain order. Note that the
    value of `order` must exist in the range of permissible values, otherwise a ValueError exception
    is thrown. Permissible values are: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35,
    41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, and 131.

    Since scipy version 1.15 included the integration scheme natively, this function just wraps
    `scipy.integrate.lebedev_rule`. The original version included with Py-optics used weight factors
    such that np.sum(w) == 1.0. Scipy uses the convention np.sum(w) = 4π. This function converts the
    returned weight factors such that np.sum(w) == 1.0, for consistency.

    Parameters
    ----------
    order : int
        Integration order to get the integration and weight factors for.

    Returns
    -------
    x, y, z, w : tuple
        Tuple of lists with the x, y and z locations of the integration coordinates, and the weight
        factors `w` to go with it.

    Raises
    ------
    NotImplementedError
        Raises an exception if `order` is not in the set of permissible values.
    """

    (x, y, z), w = lebedev_rule(order)
    return x, y, z, w / (4 * math.pi)

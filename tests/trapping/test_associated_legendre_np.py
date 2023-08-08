import pytest
import numpy as np
import numpy.polynomial as npp
from lumicks.pyoptics.trapping.associated_legendre import (
    associated_legendre,
)


def associated_legendre_npp(n, x):
    """Return the 1st order (m == 1) of the
    associated Legendre polynomial of degree n, evaluated at x [-1..1]

    This is a reference implementation in numpy, which is more generic and therefore slower.
    Furthermore, it's not compatible with Numba.
    """
    orders = np.zeros(n + 1)
    orders[n] = 1
    c = npp.Legendre(npp.legendre.legder(orders))
    return -c(x) * np.sqrt(1 - x**2)


@pytest.mark.parametrize('order', range(1, 200))
def test_legendre(order):
    x = np.linspace(-1, 1, 101)
    y1 = associated_legendre(order, x)
    y2 = associated_legendre_npp(order, x)
    np.testing.assert_allclose(y1, y2)

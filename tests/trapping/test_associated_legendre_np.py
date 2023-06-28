import pytest
import numpy as np
from lumicks.pyoptics.trapping.associated_legendre import (
    associated_legendre,
    associated_legendre_npp
)


@pytest.mark.parametrize('order', range(1, 200))
def test_legendre(order):
    x = np.linspace(-1, 1, 101)
    y1 = associated_legendre(order, x)
    y2 = associated_legendre_npp(order, x)
    np.testing.assert_allclose(y1, y2)

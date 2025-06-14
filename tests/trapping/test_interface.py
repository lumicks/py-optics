import numpy as np
import pytest

import lumicks.pyoptics.trapping.interface as ifce


@pytest.mark.parametrize("method", ("Gauss-Legendre", "Clenshaw-Curtis", "Lebedev-Laikov"))
def test_determine_integration_order_wrong_method(method):
    with pytest.raises(ValueError, match=f"Wrong type of integration method specified: {method}"):
        ifce._determine_integration_order(method, 1)


@pytest.mark.parametrize("n_orders", (-10, -1.0, -0.0001))
def test_determine_integration_order_wrong_order(n_orders):

    with pytest.raises(
        ValueError, match=f"Invalid value for n_orders. Must be > 0, got {n_orders}"
    ):
        ifce._determine_integration_order("gauss-legendre", n_orders)


@pytest.mark.parametrize("n_orders", np.arange(1, 50))
@pytest.mark.parametrize("method", ("lebedev-laikov", "gauss-legendre", "clenshaw-curtis"))
def test_determine_integration_order(method, n_orders):

    assert ifce._determine_integration_order(method, n_orders) > n_orders

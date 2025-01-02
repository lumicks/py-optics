import pytest

import lumicks.pyoptics.mathutils.lebedev_laikov as ll


@pytest.mark.parametrize("order", (-1, 0, 132.2))
def test_lebedev_laikov_args(order):
    with pytest.raises(ValueError, match=f"A value of {order} is not supported"):
        ll.get_nearest_order(order)

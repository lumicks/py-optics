import numpy as np
import pytest

import lumicks.pyoptics.mathutils.lebedev_laikov as ll


@pytest.mark.parametrize(
    "order",
    (
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
        25,
        27,
        29,
        31,
        35,
        41,
        47,
        53,
        59,
        65,
        71,
        77,
        83,
        89,
        95,
        101,
        107,
        113,
        119,
        125,
        131,
    ),
)
def test_lebedev_laikov(order):
    x, y, z, w = ll.get_integration_locations(order)
    R = np.hypot(np.hypot(x, y), z)
    np.testing.assert_allclose(R, np.ones(R.shape))
    np.testing.assert_allclose(np.sum(w), 1)


@pytest.mark.parametrize("order", (-1, 0, 132.2))
def test_lebedev_laikov_args(order):
    with pytest.raises(ValueError, match=f"A value of {order} is not supported"):
        ll.get_nearest_order(order)

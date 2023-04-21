# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pytest
import numpy as np
import pyoptics.mie_calc as mc


@pytest.mark.parametrize(
    'order', (
        3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47,
        53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131
    )
)
def test_lebedev_laikov(order):
    x, y, z, w = mc.lebedev_laikov.get_integration_locations(order)
    R = np.hypot(np.hypot(x, y), z)
    np.testing.assert_allclose(R, np.ones(R.shape))
    np.testing.assert_allclose(np.sum(w), 1)

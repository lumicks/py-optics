# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pytest
import numpy as np
from pyoptics import mie_calc as mc


@pytest.mark.parametrize('order', range(1,200))
def test_legendre(order):
    x = np.linspace(-1,1,101)
    y1 = mc.legendre.associated_legendre(order, x)
    y2 = mc.legendre.associated_legendre_npp(order, x)
    np.testing.assert_allclose(y1, y2)

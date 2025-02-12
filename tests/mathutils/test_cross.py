import numpy as np
import pytest

from lumicks.pyoptics.mathutils.integration import get_integration_locations
from lumicks.pyoptics.mathutils.vector import outer_product


@pytest.mark.parametrize(
    "e1,e2,result",
    [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((1, 0, 0), (1, 0, 0), (0, 0, 0)),
        ((0, 1, 0), (0, 1, 0), (0, 0, 0)),
        ((0, 0, 1), (0, 0, 1), (0, 0, 0)),
    ],
)
def test_cross(e1, e2, result):
    np.testing.assert_allclose(outer_product(e1, e2), result)


@pytest.mark.parametrize("order", [3, 15, 31])
def test_cross2(order: int):
    x, y, z, _ = get_integration_locations(order, "lebedev-laikov")

    rho = np.hypot(x, y)
    mask = rho > 0
    cos_phi = np.ones_like(rho)
    sin_phi = np.zeros_like(rho)
    cos_phi[mask] = x[mask] / rho[mask]
    sin_phi[mask] = y[mask] / rho[mask]
    cos_theta = z
    sin_theta = ((1 - z) * (1 + z)) ** 0.5
    er = [x, y, z]
    et = [cos_phi * cos_theta, sin_phi * cos_theta, -sin_theta]
    ep = [-sin_phi, cos_phi, np.zeros(sin_phi.shape)]
    np.testing.assert_allclose(outer_product(ep, er), et, atol=1e-7)
    np.testing.assert_allclose(outer_product(er, et), ep, atol=1e-7)
    np.testing.assert_allclose(outer_product(et, ep), er, atol=1e-7)

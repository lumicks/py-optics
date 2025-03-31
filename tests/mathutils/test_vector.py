from typing import Tuple

import numpy as np
import pytest

import lumicks.pyoptics.mathutils.vector as vector
from lumicks.pyoptics.mathutils.integration import get_integration_locations


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
    np.testing.assert_allclose(vector.outer_product(e1, e2), result)


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
    np.testing.assert_allclose(vector.outer_product(ep, er), et, atol=1e-7)
    np.testing.assert_allclose(vector.outer_product(er, et), ep, atol=1e-7)
    np.testing.assert_allclose(vector.outer_product(et, ep), er, atol=1e-7)


def test_cosines_from_unit_vectors_raises():
    with pytest.raises(
        ValueError, match="The input does not seem to be a unit vector or an array of unit vectors"
    ):
        vector.cosines_from_unit_vectors(1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "unit_vector, expected_result",
    [
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0)),
        ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 1.0)),
        ((0.0, 0.0, -1.0), (-1.0, 0.0, 1.0, 0.0)),
        ((-1.0, 0.0, 0.0), (0.0, 1.0, -1.0, 0.0)),
        ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0, -1.0)),
        ((np.sqrt(0.5), np.sqrt(0.5), 0.0), (0.0, 1.0, np.sqrt(0.5), np.sqrt(0.5))),
        ((0.0, np.sqrt(0.5), np.sqrt(0.5)), (np.sqrt(0.5), np.sqrt(0.5), 0.0, 1.0)),
        ((-np.sqrt(0.5), np.sqrt(0.5), 0.0), (0.0, 1.0, -np.sqrt(0.5), np.sqrt(0.5))),
        ((0.0, -np.sqrt(0.5), np.sqrt(0.5)), (np.sqrt(0.5), np.sqrt(0.5), 0.0, -1.0)),
        ((np.sqrt(0.5), -np.sqrt(0.5), 0.0), (0.0, 1.0, np.sqrt(0.5), -np.sqrt(0.5))),
        ((0.0, np.sqrt(0.5), -np.sqrt(0.5)), (-np.sqrt(0.5), np.sqrt(0.5), 0.0, 1.0)),
    ],
)
def test_cosines_from_unit_vectors(
    unit_vector: Tuple[float, float, float], expected_result: Tuple[float, float, float, float]
):
    cosines = vector.cosines_from_unit_vectors(*unit_vector)
    np.testing.assert_allclose(cosines, expected_result)


def test_unit_vector_from_cosines_raises():
    with pytest.raises(ValueError, match="The input does not lie on the unit circle"):
        vector.unit_vectors_from_cosines(1.0, 0.0, 0.0, 3.0)
        vector.unit_vectors_from_cosines(1.0, 2.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="The value of sin_theta cannot be less than zero."):
        vector.unit_vectors_from_cosines(0.0, -1.0, 1.0, 0.0)


@pytest.mark.parametrize(
    "function_args,expected_result",
    (
        ((1.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ((-1.0, 0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        ((0.0, 1.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
        ((0.0, 1.0, np.sqrt(0.5), np.sqrt(0.5)), (np.sqrt(0.5), np.sqrt(0.5), 0.0)),
        ((np.sqrt(0.5), np.sqrt(0.5), 0.0, 1.0), (0.0, np.sqrt(0.5), np.sqrt(0.5))),
        ((0.0, 1.0, -np.sqrt(0.5), np.sqrt(0.5)), (-np.sqrt(0.5), np.sqrt(0.5), 0.0)),
        ((np.sqrt(0.5), np.sqrt(0.5), 0.0, -1.0), (0.0, -np.sqrt(0.5), np.sqrt(0.5))),
        ((0.0, 1.0, np.sqrt(0.5), -np.sqrt(0.5)), (np.sqrt(0.5), -np.sqrt(0.5), 0.0)),
        ((-np.sqrt(0.5), np.sqrt(0.5), 0.0, 1.0), (0.0, np.sqrt(0.5), -np.sqrt(0.5))),
    ),
)
def test_unit_vectors_from_cosines(function_args, expected_result):
    unit_vectors = vector.unit_vectors_from_cosines(*function_args)
    np.testing.assert_allclose(unit_vectors, expected_result)

import numpy as np
import pytest

from lumicks.pyoptics.field_distributions.dipole import (
    farfield_dipole_position,
    field_dipole,
    field_dipole_y,
    field_dipole_z,
)
from lumicks.pyoptics.mathutils.integration import get_integration_locations, get_nearest_order


def scale_coords(coords_list, radius: float):
    return [ax * radius for ax in coords_list]


@pytest.mark.parametrize("order", [15, 28])
@pytest.mark.parametrize("radius", np.linspace(100e-9, 4e-6, 5))
def test_dipole_z(order: int, radius: float):
    n_medium = 1.33
    lambda_vac = 1064e-9
    pz = np.pi
    x, y, z, _ = get_integration_locations(get_nearest_order(order), "lebedev-laikov")
    x, y, z = scale_coords((x, y, z), radius)
    reference_fields = field_dipole([0.0, 0.0, pz], n_medium, lambda_vac, x, y, z)
    dipole_fields = field_dipole_z(pz, n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(reference_fields, dipole_fields)


@pytest.mark.parametrize("order", [15, 28])
@pytest.mark.parametrize("radius", np.linspace(100e-9, 4e-6, 5))
def test_dipole_y(order: int, radius: float):
    n_medium = 1.33
    lambda_vac = 1064e-9
    py = np.exp(1)
    x, y, z, _ = get_integration_locations(get_nearest_order(order), "lebedev-laikov")
    x, y, z = scale_coords((x, y, z), radius)
    reference_fields = field_dipole([0.0, py, 0.0], n_medium, lambda_vac, x, y, z)
    dipole_fields = field_dipole_y(py, n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(reference_fields, dipole_fields)


@pytest.mark.parametrize("order", [15, 28])
@pytest.mark.parametrize("radius", np.linspace(100e-9, 4e-6, 5))
def test_dipole_x(order: int, radius: float):
    n_medium = 1.33
    lambda_vac = 1064e-9
    py = np.exp(1)
    x, y, z, _ = get_integration_locations(get_nearest_order(order), "lebedev-laikov")
    x, y, z = scale_coords((x, y, z), radius)
    reference_fields = field_dipole([0.0, py, 0.0], n_medium, lambda_vac, x, y, z)
    dipole_fields = field_dipole_y(py, n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(reference_fields, dipole_fields)


@pytest.mark.parametrize("order", [15, 28])
@pytest.mark.parametrize("radius", np.linspace(1e-2, 1.0, 5))
def test_dipole_ff(order: int, radius: float):
    n_medium = 1.33
    lambda_vac = 1064e-9
    p = np.random.standard_normal(3)
    x, y, z, _ = get_integration_locations(get_nearest_order(order), "lebedev-laikov")
    x, y, z = scale_coords((x, y, z), radius)
    reference_fields = field_dipole(p, n_medium, lambda_vac, x, y, z, farfield=True)
    dipole_fields = farfield_dipole_position(p, n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(reference_fields[:3], dipole_fields)

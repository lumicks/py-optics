import numpy as np
import pytest
from scipy.constants import epsilon_0, mu_0
from scipy.constants import speed_of_light as C

import lumicks.pyoptics.mathutils.vector as vector
from lumicks.pyoptics.mathutils.vector import cosines_from_unit_vectors
from lumicks.pyoptics.farfield_transform.equivalence import _equivalent_currents_to_farfield
from lumicks.pyoptics.field_distributions.dipole import electric_dipole, magnetic_dipole
from lumicks.pyoptics.mathutils.integration import get_integration_locations


@pytest.mark.parametrize("order", [5, 11, 31])
@pytest.mark.parametrize("n_medium", [1.33, 1.5])
@pytest.mark.parametrize("lambda_vac", [600e-9, 1064e-9])
@pytest.mark.parametrize("component", [0, 1, 2])
def test_electric_dipoles(order, n_medium, lambda_vac, component):
    """Test that the far field due to an electric current delta distribution gives the same fields
    as an electric dipole"""
    R = 2.3e-3
    x, y, z, _ = get_integration_locations(order, method="lebedev-laikov")
    cos_theta, sin_theta, cos_phi, sin_phi = cosines_from_unit_vectors(x, y, z)
    print(cos_theta)
    x, y, z = [R * ax for ax in (x, y, z)]
    omega = 2 * np.pi * C / lambda_vac
    k = 2 * np.pi * n_medium / lambda_vac
    eta = (mu_0 / epsilon_0) ** 0.5 / n_medium
    dipole_moment = 2.2
    current_density = -1j * omega * dipole_moment
    p = [0, 0, 0]
    p[component] = dipole_moment
    J = [0, 0, 0]
    J[component] = current_density
    Ex_ref, Ey_ref, Ez_ref, _, _, _ = electric_dipole(
        p, n_medium, lambda_vac, x, y, z, farfield=True
    )
    E_theta, E_phi = _equivalent_currents_to_farfield(
        [0, 0, 0],
        J,
        [0, 0, 0],
        1.0,
        [cos_theta, sin_theta, cos_phi, sin_phi],
        R,
        np.ones_like(cos_theta, dtype=bool),
        k,
        eta,
    )
    Ex, Ey, Ez = vector.spherical_to_cartesian([x, y, z], 0.0, E_theta, E_phi)
    np.testing.assert_allclose([Ex_ref, Ey_ref, Ez_ref], [Ex, Ey, Ez])


@pytest.mark.parametrize("order", [5, 11, 31])
@pytest.mark.parametrize("n_medium", [1.3, 1.51])
@pytest.mark.parametrize("lambda_vac", [600e-9, 1064e-9])
@pytest.mark.parametrize("component", [0, 1, 2])
def test_magnetic_dipoles(order, n_medium, lambda_vac, component):
    """Test that the far field due to a magnetic current delta distribution gives the same fields
    as a magnetic dipole"""
    R = 2.3e-3
    x, y, z, _ = get_integration_locations(order, method="lebedev-laikov")
    cos_theta, sin_theta, cos_phi, sin_phi = cosines_from_unit_vectors(x, y, z)
    x, y, z = [R * ax for ax in (x, y, z)]
    omega = 2 * np.pi * C / lambda_vac
    k = 2 * np.pi * n_medium / lambda_vac
    eta = (mu_0 / epsilon_0) ** 0.5 / n_medium
    dipole_moment = 2.2  # pi a**2 I0
    current_density = -1j * omega * mu_0 * dipole_moment  # Balanis, Antenna Theory, 3rd ed. 5-21
    m = [0, 0, 0]
    m[component] = dipole_moment
    M = [0, 0, 0]
    M[component] = current_density
    Ex_ref, Ey_ref, Ez_ref, _, _, _ = magnetic_dipole(
        m, n_medium, lambda_vac, x, y, z, farfield=True
    )
    E_theta, E_phi = _equivalent_currents_to_farfield(
        [0, 0, 0],
        [0, 0, 0],
        M,
        1.0,
        [cos_theta, sin_theta, cos_phi, sin_phi],
        R,
        np.ones_like(cos_theta, dtype=bool),
        k,
        eta,
    )
    Ex, Ey, Ez = vector.spherical_to_cartesian([x, y, z], 0.0, E_theta, E_phi)

    # atol == 1024 since there is some loss of precision when converting from spherical to cartesian
    # fields: for example, for a magnetic dipole along x, the electric field is zero along x. The
    # fractions of E_theta and E_phi that point along x should be opposite in sign but equal in
    # magnitude, such that they exactly cancel. For large numbers, like here (~1e17), that is
    # impossible with normal floating point precision
    np.testing.assert_allclose([Ex, Ey, Ez], [Ex_ref, Ey_ref, Ez_ref], atol=1024)

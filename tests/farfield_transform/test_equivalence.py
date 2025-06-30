import numpy as np
import pytest
from scipy.constants import epsilon_0, mu_0
from scipy.constants import speed_of_light as C

import lumicks.pyoptics.mathutils.vector as vector
from lumicks.pyoptics.farfield_transform.equivalence import (
    NearfieldData,
    _equivalent_currents_to_farfield,
    near_field_to_far_field,
)
from lumicks.pyoptics.field_distributions.dipole import electric_dipole, magnetic_dipole
from lumicks.pyoptics.mathutils.integration import sphere as sphere_integration
from lumicks.pyoptics.mathutils.vector import cosines_from_unit_vectors
from lumicks.pyoptics.trapping import Bead


@pytest.mark.parametrize("order", [5, 11, 31])
@pytest.mark.parametrize("n_medium", [1.33, 1.5])
@pytest.mark.parametrize("lambda_vac", [600e-9, 1064e-9])
@pytest.mark.parametrize("component", [0, 1, 2])
def test_electric_dipoles(order, n_medium, lambda_vac, component):
    """Test that the far field due to an electric current delta distribution gives the same fields
    as an electric dipole"""
    R = 2.3e-3
    x, y, z, _ = sphere_integration.get_integration_locations(order, method="lebedev-laikov")
    cos_theta, sin_theta, cos_phi, sin_phi = cosines_from_unit_vectors(x, y, z)
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
    x, y, z, _ = sphere_integration.get_integration_locations(order, method="lebedev-laikov")
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


@pytest.mark.parametrize("n_medium", [1.3, 1.51])
@pytest.mark.parametrize("lambda_vac", [600e-9, 1064e-9])
@pytest.mark.parametrize(
    "dipole_moment", [(2.2, 0.0, 0.0), (0.0, np.exp(1), 0.0), (0.0, 0.0, np.pi)]
)
def test_near_field_to_far_field(n_medium, lambda_vac, dipole_moment):
    bead = Bead(lambda_vac * 2.0, n_medium, n_medium, lambda_vac)
    r_near = bead.bead_diameter / 2
    r_far = 1.0
    x, y, z, w = sphere_integration.get_integration_locations(
        sphere_integration.get_nearest_order(bead.number_of_modes, method="lebedev-laikov"),
        method="lebedev-laikov",
    )
    w = w * 4 * np.pi * r_near**2
    n = (x, y, z)
    cos_theta, sin_theta, cos_phi, sin_phi = cosines_from_unit_vectors(x, y, z)
    x_near, y_near, z_near = [r_near * ax for ax in (x, y, z)]
    x_far, y_far, z_far = [r_far * ax for ax in (x, y, z)]
    Ex_near, Ey_near, Ez_near, Hx_near, Hy_near, Hz_near = electric_dipole(
        dipole_moment, n_medium, lambda_vac, x_near, y_near, z_near
    )
    nearfield_data = NearfieldData(
        x_near,
        y_near,
        z_near,
        n,
        w,
        [Ex_near, Ey_near, Ez_near],
        [Hx_near, Hy_near, Hz_near],
        lambda_vac,
        n_medium,
    )
    E_theta, E_phi = near_field_to_far_field(
        nearfield_data,
        cos_theta,
        sin_theta,
        cos_phi,
        sin_phi,
        r_far,
    )

    Ex_far, Ey_far, Ez_far = vector.spherical_to_cartesian_from_angles(
        cos_theta, sin_theta, cos_phi, sin_phi, 0.0, E_theta, E_phi
    )

    fields_ref = electric_dipole(dipole_moment, n_medium, lambda_vac, x_far, y_far, z_far, True)
    field_max = np.max(np.abs(fields_ref))
    np.testing.assert_allclose(
        [Ex_far, Ey_far, Ez_far], fields_ref[:3], rtol=1e-4, atol=1e-8 * field_max
    )

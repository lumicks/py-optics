import numpy as np
import pytest
from scipy.constants import mu_0
from scipy.constants import speed_of_light as C

from lumicks.pyoptics.field_distributions.dipole import (
    emitted_power_magnetic_dipole,
    magnetic_dipole,
    magnetic_dipole_z,
)
from lumicks.pyoptics.mathutils.integration.sphere import (
    get_integration_locations,
    get_nearest_order,
)
from lumicks.pyoptics.mathutils.vector import outer_product


def scale_coords(coords_list, radius: float):
    return [ax * radius for ax in coords_list]


def emitted_power_kong_magnetic(m, n_medium, lambda_vac):
    """Independent implementation of emitted power by a magnetic dipole, see [1].

    ..  [1] Electromagnetic Wave Theory, Jin Au Kong, Ch. 4
    """
    # k = 2 * np.pi * n_medium / lambda_vac
    # k**4 * C * mu_0 * m**2 / (12 * np.pi * n_medium), rewritten to:
    return 4 * mu_0 * m**2 * np.pi**3 * n_medium**3 * C * lambda_vac**-4 / 3.0


def poynting(n, Ex, Ey, Ez, Hx, Hy, Hz):
    S = np.asarray(
        [
            s * nn
            for s, nn in zip(outer_product([Ex, Ey, Ez], [np.conj(H) for H in [Hx, Hy, Hz]]), n)
        ]
    )
    return S


def integrated_emitted_power(radius, w, normal, fields1):
    S = poynting(normal, *fields1)
    power = 0.5 * 4 * np.pi * radius**2 * (w * S.real).sum()
    return power


@pytest.mark.parametrize("m", [2.2, np.pi, np.exp(-1)])
@pytest.mark.parametrize("lambda_vac", [1064e-9, 600e-9])
@pytest.mark.parametrize("n_medium", [1.0, 1.33, 1.51])
def test_powers(m, lambda_vac, n_medium):
    """Test that the analytical expression for emitted power by a magnetic dipole matches an
    independent implementation"""
    assert emitted_power_magnetic_dipole(m, n_medium, lambda_vac) == pytest.approx(
        emitted_power_kong_magnetic(m, n_medium, lambda_vac)
    )


@pytest.mark.parametrize("order", [3, 15, 31])
@pytest.mark.parametrize("n_medium", [1.0, 1.33, 1.51])
@pytest.mark.parametrize("radius", [300e-9, 4e-6, 5.0])
def test_magnetic_dipole_z(order: int, n_medium: float, radius: float):
    """Test implementation from [1] against independent and an implementation from [2] that supports
    other media than vacuum by applying the duality principle to the solutions for electric dipoles
    listed in both sources. Checks fields and emitted powers.

    ..  [1] Principles of Nano-optics, 2nd Ed., Ch. 8 + application of 10.53
    ..  [2] Classical Electrodynamics, Ch. 9, 3rd Edition, J.D. Jackson
    """
    lambda_vac = 1064e-9
    mz = 3.3
    x, y, z, w = get_integration_locations(
        get_nearest_order(order, "lebedev-laikov"), "lebedev-laikov"
    )
    n = [x, y, z]
    x, y, z = scale_coords((x, y, z), radius)
    fields1 = magnetic_dipole([0.0, 0.0, mz], n_medium, lambda_vac, x, y, z, farfield=False)
    fields2 = magnetic_dipole_z(mz, n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(fields1, fields2)

    ref_power = emitted_power_magnetic_dipole(mz, n_medium, lambda_vac)
    power = integrated_emitted_power(radius, w, n, fields1)
    assert power == pytest.approx(ref_power)


@pytest.mark.parametrize("order", [3, 15, 31])
@pytest.mark.parametrize("n_medium", [1.28, 1.33, 1.51])
@pytest.mark.parametrize("radius", [300e-9, 4e-6, 5.0])
def test_magnetic_dipole(order: int, n_medium: float, radius: float):
    """Test implementation from [2] that supports
    other media than vacuum by applying the duality principle, and verify that the fields are
    invariant under rotation of the dipole axis (x, y, z). Checks fields and emitted powers.

    ..  [1] Principles of Nano-optics, 2nd Ed., Ch. 8 + application of 10.53
    ..  [2] Classical Electrodynamics, Ch. 9, 3rd Edition, J.D. Jackson
    """
    lambda_vac = 1064e-9
    mz = 3.3
    x, y, z, w = get_integration_locations(
        get_nearest_order(order, "lebedev-laikov"), "lebedev-laikov"
    )
    n = (x, y, z)
    x, y, z = scale_coords((x, y, z), radius)
    # dipole in z is benchmarked above, use result to benchmark dipole in x- and y-direction
    # Check x-dipole
    Ez, Ey, Ex, Hz, Hy, Hx = magnetic_dipole([0.0, 0.0, mz], n_medium, lambda_vac, -z, y, x)
    reference = [Ex, Ey, -Ez, Hx, Hy, -Hz]
    fields_under_test = magnetic_dipole([mz, 0.0, 0.0], n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(reference, fields_under_test)

    ref_power = emitted_power_magnetic_dipole(mz, n_medium, lambda_vac)
    power = integrated_emitted_power(radius, w, n, fields_under_test)
    assert power == pytest.approx(ref_power)

    # Check y-dipole
    Ex, Ez, Ey, Hx, Hz, Hy = magnetic_dipole([0.0, 0.0, mz], n_medium, lambda_vac, x, -z, y)
    reference = [Ex, Ey, -Ez, Hx, Hy, -Hz]
    fields_under_test = magnetic_dipole([0.0, mz, 0.0], n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(reference, fields_under_test)

    power = integrated_emitted_power(radius, w, n, fields_under_test)
    assert power == pytest.approx(ref_power)

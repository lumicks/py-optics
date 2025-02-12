import numpy as np
import pytest
from scipy.constants import epsilon_0
from scipy.constants import speed_of_light as C

from lumicks.pyoptics.field_distributions.dipole import (
    electric_dipole,
    electric_dipole_x,
    electric_dipole_y,
    electric_dipole_z,
    emitted_power_electric_dipole,
    electric_dipole_farfield_position,
)
from lumicks.pyoptics.mathutils.integration import get_integration_locations, get_nearest_order
from lumicks.pyoptics.mathutils.vector import outer_product


def scale_coords(coords_list, radius: float):
    return [ax * radius for ax in coords_list]


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


@pytest.mark.parametrize("p", [2.2, np.pi, np.exp(-1)])
@pytest.mark.parametrize("lambda_vac", [1064e-9, 600e-9])
@pytest.mark.parametrize("n_medium", [1.0, 1.33, 1.51])
def test_powers(p, lambda_vac, n_medium):
    """Test that the analytical expression for emitted power by an electric dipole matches an
    independent implementation"""

    def emitted_power_kunz_electric():
        # Lukosz, W., & Kunz, R. E. (1977). Light emission by magnetic and electric dipoles close to
        # a plane interface I Total radiated power. Journal of the Optical Society of America,
        # 67(12), 1607. doi:10.1364/josa.67.001607
        omega = 2 * np.pi * C / lambda_vac
        return abs(p) ** 2 * omega**4 * n_medium / (12 * np.pi * epsilon_0 * C**3)

    assert emitted_power_electric_dipole(p, n_medium, lambda_vac) == pytest.approx(
        emitted_power_kunz_electric()
    )


@pytest.mark.parametrize("order", [3, 15, 31])
@pytest.mark.parametrize("n_medium", [1.33, 1.51])
@pytest.mark.parametrize("radius", [300e-9, 4e-6, 5.0])
def test_electric_dipole(order: int, n_medium: float, radius: float):
    """Test the dipole field from [1] for being consistent with itself when swapping axes, that is,
     the fields for a dipole along x should be the same as for a dipole along y and z, after
     permuting the axes accordingly. The implementation from [1] is modified to support other media
     than only vacuum. Emitted power obtained through numerical integration has to match the
     analytically determined amount.

    ..  [1] Classical Electrodynamics, Ch. 9, 3rd Edition, J.D. Jackson
    """
    lambda_vac = 1064e-9
    pz = 3.3
    x, y, z, w = get_integration_locations(get_nearest_order(order), "lebedev-laikov")
    normals = (x, y, z)
    x, y, z = scale_coords((x, y, z), radius)
    # Check x-dipole is the same as z dipole, after appropriate coordinate transformations
    Ez, Ey, Ex, Hz, Hy, Hx = electric_dipole([0.0, 0.0, pz], n_medium, lambda_vac, -z, y, x)
    reference = [Ex, Ey, -Ez, Hx, Hy, -Hz]
    fields_x = electric_dipole([pz, 0.0, 0.0], n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(fields_x, reference)

    power = integrated_emitted_power(radius, w, normals, fields_x)
    assert power == pytest.approx(emitted_power_electric_dipole(pz, n_medium, lambda_vac))

    # Check y-dipole
    Ex, Ez, Ey, Hx, Hz, Hy = electric_dipole([0.0, 0.0, pz], n_medium, lambda_vac, x, -z, y)
    reference = [Ex, Ey, -Ez, Hx, Hy, -Hz]
    fields_y = electric_dipole([0.0, pz, 0.0], n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(fields_y, reference)

    power = integrated_emitted_power(radius, w, normals, fields_x)
    assert power == pytest.approx(emitted_power_electric_dipole(pz, n_medium, lambda_vac))


@pytest.mark.parametrize("order", [7, 15])
@pytest.mark.parametrize("n_medium", [1.33, 1.57])
@pytest.mark.parametrize("radius", np.linspace(100e-9, 4e-6, 5))
def test_electric_dipole_x(order: int, n_medium: float, radius: float):
    """Test implementation from [1] against independent and tweaked implementation from [2] that
    supports other media than vacuum. Checks fields and emitted powers.

    ..  [1] Principles of Nano-optics, 2nd Ed., Ch. 8
    ..  [2] Classical Electrodynamics, Ch. 9, 3rd Edition, J.D. Jackson
    """
    lambda_vac = 1064e-9
    px = 3.3
    x, y, z, w = get_integration_locations(get_nearest_order(order), "lebedev-laikov")
    n = [x, y, z]
    x, y, z = scale_coords((x, y, z), radius)
    fields1 = electric_dipole([px, 0.0, 0.0], n_medium, lambda_vac, x, y, z)
    fields2 = electric_dipole_x(px, n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(fields1, fields2)

    ref_power = emitted_power_electric_dipole(px, n_medium, lambda_vac)
    power = integrated_emitted_power(radius, w, n, fields1)
    assert power == pytest.approx(ref_power)


@pytest.mark.parametrize("order", [15, 28])
@pytest.mark.parametrize("n_medium", [1.33, 1.57])
@pytest.mark.parametrize("radius", np.linspace(100e-9, 4e-6, 5))
def test_electric_dipole_y(order: int, n_medium: float, radius: float):
    """Test implementation from [1] against tweaked implementation from [2] that supports other
    media than vacuum. Checks fields and emitted powers.

    ..  [1] Antenna Theory, Ch. 4, 3rd Edition, C. A. Balanis
    ..  [2] Classical Electrodynamics, Ch. 9, 3rd Edition, J.D. Jackson
    """
    lambda_vac = 1064e-9
    py = np.exp(1)
    x, y, z, w = get_integration_locations(get_nearest_order(order), "lebedev-laikov")
    n = [x, y, z]
    x, y, z = scale_coords((x, y, z), radius)
    fields1 = electric_dipole([0.0, py, 0.0], n_medium, lambda_vac, x, y, z)
    fields2 = electric_dipole_y(py, n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(fields1, fields2)

    ref_power = emitted_power_electric_dipole(py, n_medium, lambda_vac)
    power = integrated_emitted_power(radius, w, n, fields1)
    assert power == pytest.approx(ref_power)


@pytest.mark.parametrize("order", [3, 5, 11])
@pytest.mark.parametrize("n_medium", [1.33, 1.57])
@pytest.mark.parametrize("radius", np.linspace(1e-6, 4e-6, 4))
def test_electric_dipole_z(order: int, n_medium: float, radius: float):
    """Test implementation from [1] against tweaked implementation from [2] that supports other
    media than vacuum. Checks fields and emitted powers.

    ..  [1] Antenna Theory, Ch. 4, 3rd Edition, C. A. Balanis
    ..  [2] Classical Electrodynamics, Ch. 9, 3rd Edition, J.D. Jackson
    """
    lambda_vac = 1064e-9
    pz = 2.2
    x, y, z, w = get_integration_locations(get_nearest_order(order), "lebedev-laikov")
    n = [x, y, z]
    x, y, z = scale_coords((x, y, z), radius)
    fields1 = electric_dipole([0.0, 0.0, pz], n_medium, lambda_vac, x, y, z)
    fields2 = electric_dipole_z(pz, n_medium, lambda_vac, x, y, z)

    # Check that the fields of two independent implementations match
    np.testing.assert_allclose(fields1, fields2)

    ref_power = emitted_power_electric_dipole(pz, n_medium, lambda_vac)
    power = integrated_emitted_power(radius, w, n, fields1)
    assert power == pytest.approx(ref_power)


@pytest.mark.parametrize("order", [15, 28])
@pytest.mark.parametrize("n_medium", [1.55, 2.2])
@pytest.mark.parametrize("radius", np.linspace(1e-2, 1.0, 5))
def test_electric_dipole_ff(order: int, n_medium: float, radius: float):
    """Test farfieldimplementation from [1] against tweaked implementation from [2] that supports
    other media than vacuum. Checks fields and emitted powers.

    ..  [1] Principles of Nano-optics, 2nd Ed., Ch. 8
    ..  [2] Classical Electrodynamics, Ch. 9, 3rd Edition, J.D. Jackson
    """
    lambda_vac = 1064e-9
    p = np.random.standard_normal(3)
    x, y, z, _ = get_integration_locations(get_nearest_order(order), "lebedev-laikov")
    x, y, z = scale_coords((x, y, z), radius)
    reference_fields = electric_dipole(p, n_medium, lambda_vac, x, y, z, farfield=True)
    dipole_fields = electric_dipole_farfield_position(p, n_medium, lambda_vac, x, y, z)
    np.testing.assert_allclose(reference_fields[:3], dipole_fields)

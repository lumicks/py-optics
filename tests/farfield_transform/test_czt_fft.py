import math

import numpy as np
import pytest

from lumicks.pyoptics.farfield_transform import czt_nf_to_ff
from lumicks.pyoptics.field_distributions.dipole import electric_dipole


def create_2d_window(n: int, gamma: float = 0.0):
    k = np.linspace(-1.0, 1.0, n)
    k, l = np.meshgrid(k, k)
    r = np.hypot(k, l)
    window = np.cos(np.pi / 2.0 * (r - gamma) / (1 - gamma)) ** 2
    window[r > 1.0] = 0
    window[r < gamma] = 1.0
    return window


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_czt_fft(axis: int):
    p = [0, 0, 0]
    p[axis] = 1e-10
    lambda_vac = 630e-9
    xy_span = 80e-6
    z_distance = 2 * lambda_vac
    farfield_radius = 5.0e-3
    n_points = math.ceil(xy_span / lambda_vac) * 2 * 5
    xy = np.linspace(-xy_span / 2, xy_span / 2, n_points)
    X, Y = np.meshgrid(xy, xy)
    Z = np.ones_like(X) * z_distance

    near_fields = electric_dipole(
        p, 1.33, lambda_vac, X.reshape(X.size, -1), Y.reshape(Y.size, -1), Z.reshape(Z.size, -1)
    )
    window = create_2d_window(n_points, 0.1)

    half_angle = math.pi / 2
    Ex, Ey, Ez = [E.reshape(X.shape) * window for E in near_fields[:3]]
    sx, sy, sz, Ex_ff, Ey_ff, Ez_ff = czt_nf_to_ff(
        Ex,
        Ey,
        Ez,
        xy_span / (n_points - 1),
        farfield_radius,
        half_angle,
        lambda_vac,
        1.33,
        z_distance,
        101,
    )
    x_far, y_far, z_far = [s[sz.imag == 0.0].real * farfield_radius for s in (sx, sy, sz)]
    ref = electric_dipole(p, 1.33, lambda_vac, x_far, y_far, z_far, farfield=True)
    Ex_ref, Ey_ref, Ez_ref = [np.zeros_like(sz, dtype="complex128") for _ in range(3)]
    for source, dest in zip(ref[:3], (Ex_ref, Ey_ref, Ez_ref)):
        dest[sz.imag == 0] = source

    # Only test away from edge, where ringing causes issues with the comparison
    test_region = np.logical_and(sz.real > 0.20 * math.sin(half_angle), sz.imag == 0)
    mask = np.zeros_like(sx, dtype=bool)
    mask[test_region] = True

    for field, ref, ax in zip((Ex_ff, Ey_ff, Ez_ff), (Ex_ref, Ey_ref, Ez_ref), "xyz"):
        # Deal with quirk for a z-oriented dipole's field in z: there's an exact zero at the center,
        # but because of spectral leakage due to windowing, the CZT version of the far feld does not
        # reproduce it:
        atol = 4e11 if axis == 2 and ax == "z" else 2000
        np.testing.assert_allclose(ref.real * mask, field.real * mask, rtol=1e-1, atol=atol)
        np.testing.assert_allclose(ref.imag * mask, field.imag * mask, rtol=1e-1, atol=atol)

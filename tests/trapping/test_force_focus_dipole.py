"""This pytest compares the forces in a focus of a laser, as obtained with the dipole approximation,
with the full Mie solution"""

from itertools import product

import lumicks.pyoptics.psf as psf
import lumicks.pyoptics.trapping as trp
import numpy as np
import pytest
from scipy.constants import epsilon_0 as _EPS0

n_medium = 1.33
n_bead = 5
bead_size = 20e-9  # [m]
k = 2 * np.pi * n_medium / 1064e-9

numpoints = 5
dim = (-1e-6, 1e-6)
zrange = np.linspace(dim[0], dim[1], numpoints)
filling_factor = 0.9
NA = 1.2
focal_length = 4.43e-3
n_bfp = 1.0
bfp_sampling_n = 11
w0 = filling_factor * focal_length * NA / n_medium
objective = trp.Objective(NA=NA, focal_length=focal_length, n_medium=n_medium, n_bfp=n_bfp)
bead = trp.Bead(bead_size, n_bead, n_medium, 1064e-9)
keyword_args = {
    "lambda_vac": 1064e-9,
    "n_bfp": 1.0,
    "n_medium": n_medium,
    "focal_length": 4.43e-3,
    "NA": 1.2,
    "x_range": dim,
    "numpoints_x": numpoints,
    "y_range": dim,
    "numpoints_y": numpoints,
    "bfp_sampling_n": bfp_sampling_n,
}

# quasi-static polarizability
a_s = (4 * np.pi * _EPS0 * n_medium**2 * (bead_size / 2) ** 3 * (n_bead**2 - n_medium**2)) / (
    n_bead**2 + 2 * n_medium**2
)

# correct for radiation reaction
a = a_s + 1j * k**3 / (6 * np.pi * _EPS0 * n_medium**2) * a_s**2


def get_angles(aperture, x_bfp, y_bfp, r_bfp, bfp_sampling_n):
    sin_theta = r_bfp / focal_length
    cos_theta = np.ones_like(sin_theta)
    cos_theta[aperture] = (1 - sin_theta[aperture] ** 2) ** 0.5
    region = sin_theta > 0
    cos_phi = np.empty_like(sin_theta)
    sin_phi = np.empty_like(sin_theta)
    cos_phi[region] = x_bfp[region] / (focal_length * sin_theta[region])
    cos_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 1
    sin_phi[region] = y_bfp[region] / (focal_length * sin_theta[region])
    sin_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 0
    sin_phi[np.logical_not(aperture)] = 0
    cos_phi[np.logical_not(aperture)] = 1
    return cos_theta, sin_theta, cos_phi, sin_phi


def field_func(_, x_bfp, y_bfp, *args):
    Ein = np.exp(-((x_bfp) ** 2 + y_bfp**2) / w0**2)
    return (Ein, None)


def field_func_kx(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n):
    # Takes the derivative of the fields to X
    _, sin_theta, cos_phi, __ = get_angles(aperture, x_bfp, y_bfp, r_bfp, bfp_sampling_n)
    k = 2 * np.pi * n_medium / 1064e-9
    Kp = k * sin_theta
    Kx = -Kp * cos_phi

    Ein = np.exp(-((x_bfp) ** 2 + y_bfp**2) / w0**2) * 1j * Kx
    return (Ein, None)


def field_func_ky(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n):
    # Takes the derivative of the fields to Y
    _, sin_theta, __, sin_phi = get_angles(aperture, x_bfp, y_bfp, r_bfp, bfp_sampling_n)
    k = 2 * np.pi * n_medium / 1064e-9
    Kp = k * sin_theta
    Ky = -Kp * sin_phi

    Ein = np.exp(-((x_bfp) ** 2 + y_bfp**2) / w0**2) * 1j * Ky
    return (Ein, None)


def field_func_kz(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n):
    # Takes the derivative of the fields to Z
    cos_theta, _, __, ___ = get_angles(aperture, x_bfp, y_bfp, r_bfp, bfp_sampling_n)
    k = 2 * np.pi * n_medium / 1064e-9
    Kz = k * cos_theta

    Ein = np.exp(-((x_bfp) ** 2 + y_bfp**2) / w0**2) * 1j * Kz
    return (Ein, None)


@pytest.mark.parametrize("z_pos", zrange)
def test_force_focus(z_pos):
    Ex, Ey, Ez, X, Y, Z = psf.fast_psf(field_func, z=z_pos, return_grid=True, **keyword_args)
    Exdx, Eydx, Ezdx = psf.fast_psf(field_func_kx, z=z_pos, return_grid=False, **keyword_args)
    Exdy, Eydy, Ezdy = psf.fast_psf(field_func_ky, z=z_pos, return_grid=False, **keyword_args)
    Exdz, Eydz, Ezdz = psf.fast_psf(field_func_kz, z=z_pos, return_grid=False, **keyword_args)

    E_grad_E_x = np.conj(Ex) * Exdx + np.conj(Ey) * Eydx + np.conj(Ez) * Ezdx
    E_grad_E_y = np.conj(Ex) * Exdy + np.conj(Ey) * Eydy + np.conj(Ez) * Ezdy
    E_grad_E_z = np.conj(Ex) * Exdz + np.conj(Ey) * Eydz + np.conj(Ez) * Ezdz
    Fx = np.real(a) / 2 * E_grad_E_x.real + np.imag(a) / 2 * E_grad_E_x.imag
    Fy = np.real(a) / 2 * E_grad_E_y.real + np.imag(a) / 2 * E_grad_E_y.imag
    Fz = np.real(a) / 2 * E_grad_E_z.real + np.imag(a) / 2 * E_grad_E_z.imag

    Fx_mie = np.empty(Ex.shape)
    Fy_mie = np.empty(Ex.shape)
    Fz_mie = np.empty(Ex.shape)

    force_func = trp.force_factory(
        field_func,
        objective,
        bead=bead,
        bfp_sampling_n=bfp_sampling_n,
        num_orders=None,
        integration_orders=None,
    )
    for p in range(numpoints):
        for m in range(numpoints):
            F = force_func(
                bead_center=(X[p, m], Y[p, m], z_pos),
            )
            Fx_mie[p, m] = F[0]
            Fy_mie[p, m] = F[1]
            Fz_mie[p, m] = F[2]
    np.testing.assert_allclose(Fx, Fx_mie, rtol=1e-2, atol=1e-23)
    np.testing.assert_allclose(Fy, Fy_mie, rtol=1e-2, atol=1e-23)
    np.testing.assert_allclose(Fz, Fz_mie, rtol=1e-2, atol=1e-23)

    bead_positions = [
        (X[p, m], Y[p, m], z_pos) for p, m in product(range(numpoints), range(numpoints))
    ]
    F = trp.forces_focus(
        field_func,
        objective,
        bead=bead,
        bead_center=bead_positions,
        bfp_sampling_n=bfp_sampling_n,
        num_orders=None,
        integration_orders=None,
    )
    Fx_mie, Fy_mie, Fz_mie = [F[:, idx].reshape(numpoints, numpoints) for idx in range(3)]
    [
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=5e-24)
        for a, b in ((Fx, Fx_mie), (Fy, Fy_mie), (Fz, Fz_mie))
    ]

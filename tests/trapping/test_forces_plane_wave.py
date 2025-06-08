"""Test that the force on a bead, as exerted by a plane wave, corresponds with the analytical
expectations, for various angles and spherical integration methods."""

from functools import partial

import numpy as np
import pytest
from scipy.constants import epsilon_0 as _EPS0

import lumicks.pyoptics.trapping as trp


@pytest.mark.parametrize("method", ("lebedev-laikov", "gauss-legendre", "clenshaw-curtis"))
@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
def test_plane_wave_forces_bfp(
    method, focal_length, n_medium, NA, n_bfp=1.0, bfp_sampling_n=3, lambda_vac=1064e-9
):
    """
    Test the numerically obtained force on a bead, exerted by a plane wave,
    against the theoretically expected value that is based solely on the
    evaluation of the scattering coefficients.
    """
    objective = trp.Objective(focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium, NA=NA)

    coords, fields = objective.sample_back_focal_plane(None, bfp_sampling_n, method="equidistant")
    farfield = objective.back_focal_plane_to_farfield(coords, fields, lambda_vac)

    n_bead = 2.1
    bead_size = 1e-6  # larger than dipole approximation is valid for
    E0 = 2.2
    bead = trp.Bead(bead_size, n_bead, n_medium, lambda_vac)
    num_orders = int(0.8 * bead.number_of_orders)  # Speed up testing a bit
    Fpr = (
        bead.pressure_eff(num_orders)  # Qpr
        * (np.pi * bead.bead_diameter**2 / 4)  # Area
        * (0.5 * E0**2 * bead.n_medium**2 * _EPS0)  # Intensity
    )
    k = bead.k
    ks = k * NA / n_medium
    dk = ks / (bfp_sampling_n - 1)

    def input_field(coords, objective: trp.Objective, type: str):
        # Create an input field that is theta-polarized with 1 V/m after
        # refraction by the lens and propagation to the focal plane
        Ex = np.zeros_like(coords.x_bfp, dtype="complex128")
        Ey = np.zeros_like(coords.x_bfp, dtype="complex128")

        correction = (
            k
            * farfield.cos_theta[p, m]
            * (n_medium / n_bfp) ** 0.5
            * farfield.cos_theta[p, m] ** -0.5
        )
        if type == "theta":
            Expoint = farfield.cos_phi[p, m] * E0
            Eypoint = farfield.sin_phi[p, m] * E0
        else:
            Expoint = -farfield.sin_phi[p, m] * E0
            Eypoint = farfield.cos_phi[p, m] * E0

        Ex[p, m] = (Expoint * correction * 2 * np.pi) / (
            -1j * focal_length * np.exp(-1j * k * focal_length) * dk**2
        )
        Ey[p, m] = (Eypoint * correction * 2 * np.pi) / (
            -1j * focal_length * np.exp(-1j * k * focal_length) * dk**2
        )

        coords.weights[:] = 0.0
        coords.weights[p, m] = 1.0

        return (Ex, Ey)

    for p in range(coords.x_bfp.shape[0]):
        for m in range(coords.x_bfp.shape[0]):
            if coords.weights[p, m] == 0.0:
                continue
            F = trp.forces_focus(
                partial(input_field, type="theta"),
                objective=objective,
                bead=bead,
                bead_center=(0, 0, 0),
                bfp_sampling_n=bfp_sampling_n,
                num_orders=num_orders,
                method=method,
            )

            # direction of the plane wave, hence direction of the force
            nz = farfield.cos_theta[p, m]
            nx = -((1 - nz**2) ** 0.5) * farfield.cos_phi[p, m]
            ny = -((1 - nz**2) ** 0.5) * farfield.sin_phi[p, m]
            n = [nx, ny, nz]
            Fn = np.squeeze(F / np.linalg.norm(F))

            # check that the magnitude is the same as predicted for Mie
            # scattering
            np.testing.assert_allclose(Fpr, np.linalg.norm(F), rtol=1e-6, atol=1e-6)
            # check that the force direction is in the same direction as the
            # plane wave
            np.testing.assert_allclose(n, Fn, rtol=1e-8, atol=1e-6)

            F = trp.forces_focus(
                partial(input_field, type="phi"),
                objective=objective,
                bead=bead,
                bead_center=(0, 0, 0),
                bfp_sampling_n=bfp_sampling_n,
                num_orders=num_orders,
                method=method,
            )
            Fn = np.squeeze(F / np.linalg.norm(F))

            # check that the magnitude is the same as predicted for Mie
            # scattering
            np.testing.assert_allclose(Fpr, np.linalg.norm(F), rtol=1e-6, atol=1e-7)
            # check that the force direction is in the same direction as the
            # plane wave
            np.testing.assert_allclose(n, Fn, rtol=1e-8, atol=1e-7)

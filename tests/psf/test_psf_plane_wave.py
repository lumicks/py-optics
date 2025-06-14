"""Test that for each sample point on the back focal plane, a plane wave is emitted with the correct
amplitude, polarization and direction"""

from itertools import product

import numpy as np
import pytest

from lumicks.pyoptics.objective import BackFocalPlaneCoordinates, Objective
from lumicks.pyoptics.psf.quad import focus_quad


@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
def test_plane_wave(focal_length, n_medium, NA, n_bfp=1.0, bfp_sampling_n=7, lambda_vac=1064e-9):
    objective = Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

    coords, fields = objective.sample_back_focal_plane(None, bfp_sampling_n, method="equidistant")
    farfield = objective.back_focal_plane_to_farfield(coords, fields, lambda_vac)

    k = 2 * np.pi * n_medium / lambda_vac
    ks = k * objective.sin_theta_max
    dk = ks / (bfp_sampling_n - 1)

    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, 10)
    xy_eval = np.linspace(-lambda_vac, lambda_vac, 10)

    M = 2 * bfp_sampling_n - 1
    for pol, p, m in product(("theta", "phi"), range(M), range(M)):
        if farfield.weights[p, m] == 0.0:
            continue

        def input_field(coords: BackFocalPlaneCoordinates, obj: Objective):
            # Create an input field that is theta-polarized with 1 V/m after refraction by the lens
            # and propagation to the focal plane
            Ex = np.zeros(coords.x_bfp.shape, dtype="complex128")
            Ey = np.zeros(coords.x_bfp.shape, dtype="complex128")

            correction = (
                k
                * farfield.cos_theta[p, m]
                * (obj.n_medium / obj.n_bfp) ** 0.5
                * farfield.cos_theta[p, m] ** -0.5
            )

            Expoint = farfield.cos_phi[p, m] if pol == "theta" else -farfield.sin_phi[p, m]
            Eypoint = farfield.sin_phi[p, m] if pol == "theta" else farfield.cos_phi[p, m]
            Ex[p, m] = (Expoint * correction * 2 * np.pi) / (
                -1j * focal_length * np.exp(-1j * k * focal_length) * dk**2
            )
            Ey[p, m] = (Eypoint * correction * 2 * np.pi) / (
                -1j * focal_length * np.exp(-1j * k * focal_length) * dk**2
            )
            return (Ex, Ey)

        Ex, Ey, Ez, X, Y, _ = focus_quad(
            input_field,
            objective,
            lambda_vac=lambda_vac,
            x=xy_eval,
            y=xy_eval,
            z=z_eval,
            integration_order=bfp_sampling_n,
            return_grid=True,
            integration_method="equidistant",
        )

        kz = k * farfield.cos_theta[p, m]
        kx = -k * farfield.sin_theta[p, m] * farfield.cos_phi[p, m]
        ky = -k * farfield.sin_theta[p, m] * farfield.sin_phi[p, m]

        # Check convention, +1j for k vector as we use -1j for time phasor
        Exp = np.exp(1j * (kx * X + ky * Y + kz * z_eval))

        if pol == "theta":
            Expw = farfield.cos_theta[p, m] * farfield.cos_phi[p, m] * Exp
            Eypw = farfield.cos_theta[p, m] * farfield.sin_phi[p, m] * Exp
            Ezpw = farfield.sin_theta[p, m] * Exp

            np.testing.assert_allclose([Ex, Ey, Ez], [Expw, Eypw, Ezpw])
            np.testing.assert_allclose(
                np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2, np.ones(Ex.shape)
            )
        else:
            Expw = -farfield.sin_phi[p, m] * Exp
            Eypw = farfield.cos_phi[p, m] * Exp
            Ezpw = np.zeros(Ex.shape) * Exp
            # Ezpw == 0, but Ez has some numerical rounding errors. Therefore specify an absolute
            # tolerance that is acceptable
            np.testing.assert_allclose([Ex, Ey, Ez], [Expw, Eypw, Ezpw], atol=1e-14)
            np.testing.assert_allclose(
                np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2, np.ones(Ex.shape)
            )

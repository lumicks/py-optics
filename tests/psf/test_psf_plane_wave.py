import numpy as np
import pytest
from itertools import product
from lumicks.pyoptics.psf.direct import direct_psf
from lumicks.pyoptics.trapping.objective import Objective


@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
def test_plane_wave(focal_length, n_medium, NA, n_bfp=1.0, bfp_sampling_n=7, lambda_vac=1064e-9):
    # We don't use the class Objective in pyoptics.psf, but it is handy here to get the angles
    objective = Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

    def dummy(x_bfp, **kwargs):
        """A do-nothing function"""
        return (np.zeros_like(x_bfp), None)

    coords, fields = objective.sample_back_focal_plane(dummy, bfp_sampling_n)
    farfield = objective.back_focal_plane_to_farfield(coords, fields, lambda_vac)

    k = 2 * np.pi * n_medium / lambda_vac
    ks = k * NA / n_medium
    dk = ks / (bfp_sampling_n - 1)

    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, 10)
    xy_eval = np.linspace(-lambda_vac, lambda_vac, 10)

    M = 2 * bfp_sampling_n - 1
    for pol, p, m in product(("theta", "phi"), range(M), range(M)):
        if not farfield.aperture[p, m]:
            continue

        def input_field(_, x_bfp, *args):
            # Create an input field that is theta-polarized with 1 V/m after refraction by the
            # lens and propagation to the focal plane
            Ex = np.zeros(x_bfp.shape, dtype="complex128")
            Ey = np.zeros(x_bfp.shape, dtype="complex128")

            correction = (
                k * farfield.cos_theta[p, m]
                * (n_medium / n_bfp)**0.5 * farfield.cos_theta[p, m]**-0.5
            )  # fmt: skip

            Expoint = farfield.cos_phi[p, m] if pol == "theta" else -farfield.sin_phi[p, m]
            Eypoint = farfield.sin_phi[p, m] if pol == "theta" else farfield.cos_phi[p, m]
            Ex[p, m] = (
                Expoint * correction * 2 * np.pi
                / (-1j * focal_length * np.exp(-1j * k * focal_length) * dk**2)
            )  # fmt: skip
            Ey[p, m] = (
                Eypoint * correction * 2 * np.pi
                / (-1j * focal_length * np.exp(-1j * k * focal_length) * dk**2)
            )  # fmt: skip
            return (Ex, Ey)

        Ex, Ey, Ez, X, Y, _ = direct_psf(
            input_field,
            lambda_vac=lambda_vac,
            n_bfp=n_bfp,
            n_medium=n_medium,
            focal_length=focal_length,
            NA=NA,
            x=xy_eval,
            y=xy_eval,
            z=z_eval,
            bfp_sampling_n=bfp_sampling_n,
            return_grid=True,
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

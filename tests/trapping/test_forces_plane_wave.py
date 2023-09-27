import numpy as np
import pytest
from scipy.constants import epsilon_0 as _EPS0
import lumicks.pyoptics.trapping as trp


@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
def test_plane_wave_forces_bfp(
    focal_length, n_medium, NA, n_bfp=1.0, bfp_sampling_n=3, lambda_vac=1064e-9
):
    """
    Test the numerically obtained force on a bead, exerted by a plane wave,
    against the theoretically expected value that is based solely on the
    evaluation of the scattering coefficients.
    """
    objective = trp.Objective(focal_length=focal_length,
                             n_bfp=n_bfp, n_medium=n_medium, NA=NA)

    def dummy(_, x_bfp, *args):
        return (np.zeros_like(x_bfp), None)

    coords, fields = objective.sample_back_focal_plane(dummy, bfp_sampling_n)
    farfield = objective.back_focal_plane_to_farfield(
        coords, fields, lambda_vac)

    n_bead = 2.1
    bead_size = 1e-6  # larger than dipole approximation is valid for
    E0 = 2.2
    bead = trp.Bead(bead_size, n_bead, n_medium, lambda_vac)
    num_orders = bead.number_of_orders * 2
    Fpr = (
        bead.pressure_eff(num_orders) * np.pi * bead.bead_diameter**2 / 8 *
        E0**2 * bead.n_medium**2 * _EPS0
    )
    k = bead.k
    ks = k * NA / n_medium
    dk = ks / (bfp_sampling_n - 1)

    def input_field_Etheta(aperture, x_bfp, *args):
        # Create an input field that is theta-polarized with 1 V/m after
        # refraction by the lens and propagation to the focal plane
        Ex = np.zeros_like(x_bfp, dtype='complex128')
        Ey = np.zeros_like(x_bfp, dtype='complex128')

        correction = (
            k * farfield.cos_theta[p, m] * (n_medium / n_bfp)**0.5 *
            farfield.cos_theta[p, m]**-0.5
        )
        Expoint = farfield.cos_phi[p, m] * E0
        Eypoint = farfield.sin_phi[p, m] * E0

        Ex[p, m] = (
            Expoint * correction * 2 * np.pi /
            (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
        )
        Ey[p, m] = (
            Eypoint * correction * 2 * np.pi /
            (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
        )

        aperture[:] = False
        aperture[p, m] = True

        return (Ex, Ey)

    def input_field_Ephi(aperture, x_bfp, *args):
        # Create an input field that is phi-polarized with 1 V/m after
        # refraction by the lens and propagation to the focal plane
        Ex = np.zeros_like(x_bfp, dtype='complex128')
        Ey = np.zeros_like(x_bfp, dtype='complex128')

        correction = k * \
            farfield.cos_theta[p, m] * (n_medium / n_bfp)**0.5 * \
            farfield.cos_theta[p, m]**-0.5
        Expoint = -farfield.sin_phi[p, m] * E0
        Eypoint = farfield. cos_phi[p, m] * E0
        Ex[p, m] = (
            Expoint * correction * 2 * np.pi /
            (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
        )
        Ey[p, m] = (
            Eypoint * correction * 2 * np.pi /
            (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
        )

        aperture[:] = False
        aperture[p, m] = True

        return (Ex, Ey)

    for p in range(coords.x_bfp.shape[0]):
        for m in range(coords.x_bfp.shape[0]):

            if not coords.aperture[p, m]:
                continue
            F = trp.forces_focus(
                input_field_Etheta, objective=objective, bead=bead,
                bead_center=(0, 0, 0), bfp_sampling_n=bfp_sampling_n,
                verbose=False, num_orders=num_orders
            )

            # direction of the plane wave, hence direction of the force
            nz = farfield.cos_theta[p, m]
            nx = - (1 - nz**2)**0.5 * farfield.cos_phi[p, m]
            ny = - (1 - nz**2)**0.5 * farfield.sin_phi[p, m]
            n = [nx, ny, nz]
            Fn = np.squeeze(F/np.linalg.norm(F))

            # check that the magnitude is the same as predicted for Mie
            # scattering
            np.testing.assert_allclose(
                Fpr, np.linalg.norm(F), rtol=1e-8, atol=1e-4)
            # check that the force direction is in the same direction as the
            # plane wave
            np.testing.assert_allclose(n, Fn, rtol=1e-8, atol=1e-4)

            F = trp.forces_focus(
                input_field_Ephi, objective=objective, bead=bead,
                bead_center=(0, 0, 0), bfp_sampling_n=bfp_sampling_n,
                verbose=False, num_orders=num_orders
            )
            Fn = np.squeeze(F/np.linalg.norm(F))

            # check that the magnitude is the same as predicted for Mie
            # scattering
            np.testing.assert_allclose(
                Fpr, np.linalg.norm(F), rtol=1e-8, atol=1e-4)
            # check that the force direction is in the same direction as the
            # plane wave
            np.testing.assert_allclose(n, Fn, rtol=1e-8, atol=1e-4)

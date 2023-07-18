import numpy as np
import pytest
import lumicks.pyoptics.trapping as trp


@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("n_angles", [7, 11])
def test_plane_wave_direct(
    n_medium, NA, n_angles, lambda_vac=1064e-9
):
    """Test to make sure trapping.fields_plane_wave() returns a plane wave.

    Parameters
    ----------
    n_medium : float
        refractive index of the medium [-]
    NA : float
        numerical aperture of the objective [-]
    n_angles : int
        number of angles to try.
    lambda_vac : float, optional
        wavelength in vacuum, by default 1064e-9 [m]
    """
    k = 2*np.pi*n_medium / lambda_vac

    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, 21)
    xy_eval = np.linspace(-lambda_vac, lambda_vac, 21)

    sin_th_max = NA / n_medium
    sin_th = np.linspace(-sin_th_max, sin_th_max, num=n_angles)
    x_bfp, y_bfp = np.meshgrid(sin_th, sin_th)
    r_bfp = np.hypot(x_bfp, y_bfp)
    aperture = r_bfp < sin_th_max
    theta = np.zeros(r_bfp.shape)
    theta[aperture] = np.arcsin(r_bfp[aperture])

    phi = np.arctan2(y_bfp, x_bfp)
    phi[np.logical_not(aperture)] = 0
    phi[r_bfp == 0] = 0

    for p in range(n_angles):
        for m in range(n_angles):
            if not aperture[p, m]:
                continue
            bead = trp.Bead(1e-9, n_medium, n_medium, lambda_vac)
            Ex, Ey, Ez, X, Y, Z = trp.fields_plane_wave(
                bead, x=xy_eval, y=xy_eval, z=z_eval,
                theta=theta[p, m], phi=phi[p, m],
                polarization=(1, 0), return_grid=True, verbose=False
            )
            kz = k * np.cos(theta[p, m])
            kx = - k * np.sin(theta[p, m]) * np.cos(phi[p, m])
            ky = - k * np.sin(theta[p, m]) * np.sin(phi[p, m])

            # Check convention, +1j for k vector as we use -1j for time phasor
            Exp = np.exp(1j * (kx * X + ky * Y + kz * z_eval))
            Expw = np.cos(theta[p, m]) * np.cos(phi[p, m]) * Exp
            Eypw = np.cos(theta[p, m]) * np.sin(phi[p, m]) * Exp
            Ezpw = np.sin(theta[p, m]) * Exp

            # return [Expw, Eypw, Ezpw, Ex, Ey, Ez]
            np.testing.assert_allclose(Ex, Expw, atol=1e-14)
            np.testing.assert_allclose(Ey, Eypw, atol=1e-14)
            np.testing.assert_allclose(Ez, Ezpw, atol=1e-14)
            np.testing.assert_allclose(
                np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2,
                np.ones(Ex.shape)
            )

            Ex, Ey, Ez, X, Y, Z = trp.fields_plane_wave(
                bead, x=xy_eval, y=xy_eval, z=z_eval,
                theta=theta[p, m], phi=phi[p, m],
                polarization=(0, 1), return_grid=True, verbose=False
            )
            Expw = -np.sin(phi[p, m]) * Exp
            Eypw = np.cos(phi[p, m]) * Exp
            Ezpw = np.zeros(Ex.shape) * Exp

            # Ezpw == 0, but Ez has some numerical rounding errors. Therefore
            # specify an absolute tolerance that is acceptable
            np.testing.assert_allclose(Ex, Expw, atol=1e-14)
            np.testing.assert_allclose(Ey, Eypw, atol=1e-14)
            np.testing.assert_allclose(Ez, Ezpw, atol=1e-14)
            np.testing.assert_allclose(
                np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2,
                np.ones(Ex.shape)
            )


@pytest.mark.parametrize("n_medium, NA", [(1.0, 0.9), (1.33, 1.2), (1.5, 1.4)])
@pytest.mark.parametrize("focal_length", [4.43e-3, 6e-3])
def test_plane_wave_bfp(
    focal_length, n_medium, NA, n_bfp=1.0, bfp_sampling_n=4, lambda_vac=1064e-9
):
    """Test to make sure that each sample of the back focal plane generates a plane wave

    Parameters
    ----------
    focal_length : float
        focal length of the objective [m]
    n_medium : float
        refractive index of the medium [-]
    NA : float
        numerical aperture of the objective [-]
    n_bfp : float, optional
        refractive index of the medium at the back focal plane, by default 1.0 [-]
    bfp_sampling_n : int, optional
        number of samples of the back focal plane, by default 4
    lambda_vac : float, optional
        wavelength in vacuum, by default 1064e-9 [m]

    """
    num_pts = 21
    bead = trp.Bead(bead_diameter=1e-9, n_bead=n_medium,
                   n_medium=n_medium, lambda_vac=lambda_vac)
    objective = trp.Objective(
        NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

    def dummy(x_bfp, **kwargs):
        return (np.zeros_like(x_bfp), None)

    coords, fields = objective.sample_back_focal_plane(dummy, bfp_sampling_n)
    farfield = objective.back_focal_plane_to_farfield(
        coords, fields, lambda_vac)

    k = bead.k
    ks = k * NA / n_medium
    dk = ks / (bfp_sampling_n - 1)

    z_eval = np.linspace(-2 * lambda_vac, 2 * lambda_vac, num_pts)
    xy_eval = np.linspace(-lambda_vac, lambda_vac, num_pts)

    M = 2 * bfp_sampling_n - 1
    for p in range(M):
        for m in range(M):
            if not farfield.aperture[p, m]:
                continue

            def input_field_Etheta(x_bfp, **kwargs):
                # Create an input field that is theta-polarized with 1 V/m
                # after refraction by the lens and propagation to the focal
                # plane

                Ex = np.zeros_like(x_bfp, dtype='complex128')
                Ey = np.zeros_like(x_bfp, dtype='complex128')

                correction = farfield.kz[p, m] * (
                    np.sqrt(n_bfp / n_medium) *
                    np.sqrt(farfield.cos_theta[p, m])
                )**-1
                Expoint = farfield.cos_phi[p, m]
                Eypoint = farfield.sin_phi[p, m]

                phase = (-1j * objective.focal_length * (
                    np.exp(-1j * bead.k * objective.focal_length) *
                    dk**2 / (2 * np.pi)
                ))**-1
                Ex[p, m] = Expoint * correction * phase
                Ey[p, m] = Eypoint * correction * phase

                return (Ex, Ey)

            def input_field_Ephi(x_bfp, **kwargs):
                # Create an input field that is phi-polarized with 1 V/m after
                # refraction by the lens and propagation to the focal plane
                Ex = np.zeros(x_bfp.shape, dtype='complex128')
                Ey = np.zeros(x_bfp.shape, dtype='complex128')

                correction = farfield.kz[p, m] * \
                    farfield.cos_theta[p, m]**-0.5 * (n_medium/n_bfp)**0.5
                Expoint = -farfield.sin_phi[p, m]
                Eypoint = farfield.cos_phi[p, m]
                Ex[p, m] = Expoint * correction * 2 * np.pi / \
                    (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)
                Ey[p, m] = Eypoint * correction * 2 * np.pi / \
                    (-1j*focal_length*np.exp(-1j*k*focal_length)*dk**2)

                return (Ex, Ey)

            Ex, Ey, Ez, X, Y, Z = trp.fields_focus(
                input_field_Etheta, bead=bead, objective=objective,
                x=xy_eval, y=0, z=z_eval,
                bfp_sampling_n=bfp_sampling_n, return_grid=True, verbose=False
            )
            kz = farfield.kz[p, m]
            kx = farfield.kx[p, m]
            ky = farfield.ky[p, m]

            # Check convention, +1j for k vector as we use -1j for time phasor
            Exp = np.exp(1j * (kx * X + ky * Y + kz * Z))
            Expw = farfield.cos_theta[p, m] * farfield.cos_phi[p, m] * Exp
            Eypw = farfield.cos_theta[p, m] * farfield.sin_phi[p, m] * Exp
            Ezpw = (1-farfield.cos_theta[p, m]**2)**0.5 * Exp

            np.testing.assert_allclose(Ex, Expw, atol=1e-14)
            np.testing.assert_allclose(Ey, Eypw, atol=1e-14)
            np.testing.assert_allclose(Ez, Ezpw, atol=1e-14)
            np.testing.assert_allclose(
                np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2,
                np.ones(Ex.shape)
            )

            Ex, Ey, Ez, X, Y, Z = trp.fields_focus(
                input_field_Ephi, bead=bead, objective=objective,
                x=xy_eval, y=0, z=z_eval,
                bfp_sampling_n=bfp_sampling_n, return_grid=True, verbose=False
            )
            Expw = -farfield.sin_phi[p, m] * Exp
            Eypw = farfield.cos_phi[p, m] * Exp
            Ezpw = np.zeros(Ex.shape)
            # Ezpw == 0, but Ez has some numerical rounding errors. Therefore
            # specify an absolute tolerance that is acceptable
            np.testing.assert_allclose(Ex, Expw, atol=1e-14)
            np.testing.assert_allclose(Ey, Eypw, atol=1e-14)
            np.testing.assert_allclose(Ez, Ezpw, atol=1e-14)
            np.testing.assert_allclose(
                np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2,
                np.ones(Ex.shape)
            )

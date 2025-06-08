import math

import numpy as np
import pytest

from lumicks.pyoptics.objective import Objective


@pytest.mark.parametrize(
    "NA, focal_length, n_medium, n_bfp, error_msg",
    [
        (1.1, 4.43e-3, 1.0, 1.0, "The NA of the objective cannot be larger than n_medium"),
        (
            -1.9,
            4.43e-3,
            -1.0,
            1.0,
            "Only positive and real refractive indices are supported for n_bfp and n_medium",
        ),
        (
            0.9,
            4.43e-3,
            1.0,
            -1.0,
            "Only positive and real refractive indices are supported for n_bfp and n_medium",
        ),
        (
            0.9,
            4.43e-3,
            1.0 + 0.1j,
            1.0,
            "Only positive and real refractive indices are supported for n_bfp and n_medium",
        ),
        (
            0.9,
            4.43e-3,
            1.0,
            1.0 + 0.1j,
            "Only positive and real refractive indices are supported for n_bfp and n_medium",
        ),
        (1.0, 0.0, 1.0, 1.0, "focal_length needs to be strictly positive"),
        (0.0, 4.43e-3, 1.0, 1.0, "NA needs to be strictly positive and real"),
    ],
)
def test_objective_value_errors(NA, focal_length, n_medium, n_bfp, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)


@pytest.mark.parametrize(
    "NA, n_medium, wavelength", [(0.5, 1.0, 800e-9), (1.2, 1.33, 600e-9), (1.49, 1.51, 1064e-9)]
)
@pytest.mark.parametrize("max_pos", np.linspace(0.0, 100e-6, 21))
def test_integration_order_equidistant_in_plane(NA, n_medium, max_pos, wavelength):
    obj = Objective(NA=NA, focal_length=4e-3, n_bfp=1.0, n_medium=n_medium)
    assert obj.minimal_integration_order(
        (np.ones(1) * max_pos, np.zeros(1), np.zeros(1)), wavelength, method="equidistant"
    ) == max(
        math.ceil(1 + 2 * NA * max_pos / wavelength), 2
    )  # N_z is 2 at the minimum


@pytest.mark.parametrize(
    "NA, n_medium, wavelength", [(0.5, 1.0, 800e-9), (1.2, 1.33, 600e-9), (1.49, 1.51, 1064e-9)]
)
@pytest.mark.parametrize("max_pos", np.linspace(0.0, 100e-6, 21))
def test_minimal_integration_order_equidistant_out_of_plane(NA, n_medium, max_pos, wavelength):
    obj = Objective(NA=NA, focal_length=4e-3, n_bfp=1.0, n_medium=n_medium)

    if abs(max_pos) > 0.0 and (
        (c := (wavelength / (abs(max_pos) * 2 * n_medium) + (1 - obj.sin_theta_max**2) ** 0.5)) < 1
    ):
        b = NA / n_medium
        x = ((1 - c**2) / b**2) ** 0.5
        N = math.ceil((x - 2) / (x - 1))
    else:
        N = 2
    assert N == obj.minimal_integration_order((0, 0, max_pos), wavelength, method="equidistant")


def test_minimal_sampling_order_raises():
    obj = Objective(NA=1.0, focal_length=4e-3, n_bfp=1.0, n_medium=1.0)
    with pytest.raises(ValueError, match="Unsupported method: unsupported"):
        obj.minimal_integration_order(np.zeros((3, 1)), lambda_vac=500e-9, method="unsupported")

    for coordinates in [(0,), (0, 0), (0, 0, 0, 0)]:
        with pytest.raises(
            RuntimeError,
            match=f"Unexpected length of coordinates: expected 3, got {len(coordinates)}.",
        ):
            obj.minimal_integration_order(coordinates, lambda_vac=500e-9, method="equidistant")


# TODO: test integration order for method "peirce"

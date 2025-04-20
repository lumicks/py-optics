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

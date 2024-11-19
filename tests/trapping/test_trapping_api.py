"""Test the czt-based implementation against a trivial implementation which sums plane waves"""

import re

import pytest

import lumicks.pyoptics.trapping as trp

bead_diam = 1e-6
n_bead = 1.6
n_medium = 1.33
n_wrong_medium = 1.3
NA = 1.2
focal_length = 4.43e-3
n_bfp = 1.0
bead = trp.Bead(bead_diameter=bead_diam, n_bead=n_bead, n_medium=n_wrong_medium)
objective = trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)


def dummy(*args):
    pass  # We should never get there


# Test bead immersion medium and objective medium have to be the same
@pytest.mark.parametrize(
    "function",
    [
        trp.fields_focus,
        trp.force_factory,
    ],
)
def test_throw_on_wrong_medium(function) -> None:
    bead = trp.Bead(bead_diameter=bead_diam, n_bead=n_bead, n_medium=n_wrong_medium)
    objective = trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)
    with pytest.raises(
        ValueError,
        match=re.escape("The immersion medium of the bead and the objective have to be the same"),
    ):
        function(dummy, objective=objective, bead=bead)


@pytest.mark.parametrize(
    "NA, focal_length, n_medium, n_bfp, error_msg",
    [
        (1.1, 4.43e-3, 1.0, 1.0, "The NA of the objective cannot be larger than n_medium"),
        (
            -1.9,
            4.43e-3,
            -1.0,
            1.0,
            "Only positive refractive indices are supported for n_bfp and n_medium",
        ),
        (
            0.9,
            4.43e-3,
            1.0,
            -1.0,
            "Only positive refractive indices are supported for n_bfp and n_medium",
        ),
        (1.0, 0.0, 1.0, 1.0, "focal_length needs to be strictly positive"),
        (0.0, 4.43e-3, 1.0, 1.0, "NA needs to be strictly positive"),
    ],
)
def test_objective_value_errors(NA, focal_length, n_medium, n_bfp, error_msg):
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

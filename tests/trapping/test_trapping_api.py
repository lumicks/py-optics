"""Test the czt-based implementation against a trivial implementation which sums plane waves"""

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
        match="The immersion medium of the bead and the objective have to be the same",
    ):
        function(lambda: None, objective=objective, bead=bead, integration_order_bfp=3)

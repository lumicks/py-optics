"""Test the czt-based implementation against a trivial implementation which sums plane waves"""

import re

import pytest

from lumicks.pyoptics.psf.fast import fast_gauss


def test_num_points():
    with pytest.raises(ValueError, match=re.escape("numpoints_x needs to be >= 1")):
        fast_gauss(1, 1, 1, 1, 1, 1, (-1, 1), -1, (-1, 1), 2, 0)

    with pytest.raises(ValueError, match=re.escape("numpoints_y needs to be >= 1")):
        fast_gauss(1, 1, 1, 1, 1, 1, (-1, 1), 2, (-1, 1), -1, 0)


def test_range():
    with pytest.raises(
        RuntimeError, match=re.escape("x_range needs to be a float or a (min, max) tuple")
    ):
        fast_gauss(1, 1, 1, 1, 1, 1, (-1, 1, 2), 3, (-1, 1), 1, 0)

    with pytest.raises(
        RuntimeError, match=re.escape("y_range needs to be a float or a (min, max) tuple")
    ):
        fast_gauss(1, 1, 1, 1, 1, 1, (-1, 1), 3, (-1, 1, 2, 4), 1, 0)


def test_range_and_points():
    with pytest.raises(
        ValueError, match=re.escape("x_range needs to be a location (float) for numpoints_x == 1")
    ):
        fast_gauss(1, 1, 1, 1, 1, 1, (-1, 1), 1, (-1, 1), 10, 0)

    with pytest.raises(
        ValueError, match=re.escape("y_range needs to be a location (float) for numpoints_y == 1")
    ):
        fast_gauss(1, 1, 1, 1, 1, 1, (-1, 1), 3, (-1, 1), 1, 0)

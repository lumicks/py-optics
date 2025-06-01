"""Test that exceptions are raised as required"""

import re

import pytest

from lumicks.pyoptics.psf.czt import focus_czt


def test_num_points():
    with pytest.raises(ValueError, match=re.escape("numpoints_x needs to be >= 1")):
        focus_czt(1, 1, 1, (-1, 1), -1, (-1, 1), 2, 0)

    with pytest.raises(ValueError, match=re.escape("numpoints_y needs to be >= 1")):
        focus_czt(1, 1, 1, (-1, 1), 2, (-1, 1), -1, 0)


def test_range():
    with pytest.raises(
        RuntimeError, match=re.escape("x_range needs to be a float or a (min, max) tuple")
    ):
        focus_czt(1, 1, 1, (-1, 1, 2), 3, (-1, 1), 1, 0)

    with pytest.raises(
        RuntimeError, match=re.escape("y_range needs to be a float or a (min, max) tuple")
    ):
        focus_czt(1, 1, 1, (-1, 1), 3, (-1, 1, 2, 4), 1, 0)


def test_range_and_points():
    with pytest.raises(
        ValueError, match=re.escape("x_range needs to be a location (float) for numpoints_x == 1")
    ):
        focus_czt(1, 1, 1, (-1, 1), 1, (-1, 1), 10, 0)

    with pytest.raises(
        ValueError, match=re.escape("y_range needs to be a location (float) for numpoints_y == 1")
    ):
        focus_czt(1, 1, 1, (-1, 1), 3, (-1, 1), 1, 0)

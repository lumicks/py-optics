from numba import config

from ..objective import Objective
from .bead import Bead
from .interface import (
    absorbed_power_focus,
    fields_focus,
    fields_focus_gaussian,
    fields_plane_wave,
    force_factory,
    forces_focus,
    scattered_power_focus,
)

config.THREADING_LAYER = "threadsafe"

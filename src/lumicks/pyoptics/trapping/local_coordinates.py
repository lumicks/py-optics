import numpy as np
from enum import Enum


class CoordLocation(Enum):
    INSIDE_BEAD = 0
    OUTSIDE_BEAD = 1
    EVERYWHERE = 2


class Coordinates:
    @property
    def coordinate_shape(self):
        raise RuntimeError("Cannot instantiate base class Coordinates")

    @property
    def r(self):
        raise RuntimeError("Cannot instantiate base class Coordinates")

    @property
    def region(self):
        raise RuntimeError("Cannot instantiate base class Coordinates")

    @property
    def xyz_stacked(self):
        raise RuntimeError("Cannot instantiate base class Coordinates")


class LocalBeadCoordinates(Coordinates):
    __slots__ = (
        "_bead_diameter",
        "_outside_bead",
        "_inside_bead",
        "_x_local",
        "_y_local",
        "_z_local",
        "_r",
        "_xyz_shape",
    )

    def __init__(self, x, y, z, bead_diameter, bead_center=(0, 0, 0), grid=True):
        """Set up local coordinate system around bead"""

        self._bead_diameter = bead_diameter
        # Set up a meshgrid, or take the list of x, y and z points as
        # coordinates
        if grid:
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        else:
            assert x.size == y.size == z.size, "x, y and z need to be of the same length"
            X = np.atleast_3d(x)
            Y = np.atleast_3d(y)
            Z = np.atleast_3d(z)

        # Store for rearranging the coordinates to original format
        self._xyz_shape = X.shape

        # Local coordinate system around the bead and make it a vector
        self._x_local = X.reshape((1, -1)) - bead_center[0]
        self._y_local = Y.reshape((1, -1)) - bead_center[1]
        self._z_local = Z.reshape((1, -1)) - bead_center[2]

        # Calculate the distance of a coordinate to the center of the bead
        self._r = np.hypot(np.hypot(self._x_local, self._y_local), self._z_local)
        self._outside_bead = self._r > bead_diameter / 2
        self._inside_bead = self._r <= bead_diameter / 2

    def get_xyz_stacked(self, location: CoordLocation):
        if location == CoordLocation.INSIDE_BEAD:
            return np.vstack((self._x_inside, self._y_inside, self._z_inside))
        if location == CoordLocation.OUTSIDE_BEAD:
            return np.vstack((self._x_outside, self._y_outside, self._z_outside))
        if location == CoordLocation.EVERYWHERE:
            return np.vstack((self._x_local, self._y_local, self._z_local))
        else:
            raise ValueError("Unsupported location for coordinates given")

    @property
    def _r_inside(self):
        return self._r[self._inside_bead]

    @property
    def _r_outside(self):
        return self._r[self._outside_bead]

    @property
    def _x_inside(self):
        return self._x_local[self._inside_bead]

    @property
    def _y_inside(self):
        return self._y_local[self._inside_bead]

    @property
    def _z_inside(self):
        return self._z_local[self._inside_bead]

    @property
    def _x_outside(self):
        return self._x_local[self._outside_bead]

    @property
    def _y_outside(self):
        return self._y_local[self._outside_bead]

    @property
    def _z_outside(self):
        return self._z_local[self._outside_bead]

    @property
    def _region_inside_bead(self):
        return self._inside_bead

    @property
    def _region_outside_bead(self):
        return self._outside_bead

    @property
    def coordinate_shape(self):
        return self._xyz_shape

    @property
    def xyz_stacked(self):
        return self.get_xyz_stacked(CoordLocation.EVERYWHERE)


class InternalBeadCoordinates(Coordinates):
    def __init__(self, local_coordinates: LocalBeadCoordinates) -> None:
        self._local_coordinates = local_coordinates

    @property
    def xyz_stacked(self):
        return self._local_coordinates.get_xyz_stacked(CoordLocation.INSIDE_BEAD)

    @property
    def r(self):
        return self._local_coordinates._r_inside

    @property
    def region(self):
        return self._local_coordinates._region_inside_bead

    @property
    def coordinate_shape(self):
        return self._local_coordinates.coordinate_shape


class ExternalBeadCoordinates(Coordinates):
    def __init__(self, local_coordinates: LocalBeadCoordinates) -> None:
        self._local_coordinates = local_coordinates

    @property
    def xyz_stacked(self):
        return self._local_coordinates.get_xyz_stacked(CoordLocation.OUTSIDE_BEAD)

    @property
    def r(self):
        return self._local_coordinates._r_outside

    @property
    def region(self):
        return self._local_coordinates._region_outside_bead

    @property
    def coordinate_shape(self):
        return self._local_coordinates.coordinate_shape

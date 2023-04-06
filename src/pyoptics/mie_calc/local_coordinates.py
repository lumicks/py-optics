import numpy as np

class LocalBeadCoordinates:

    __slots__ = (
        '_bead_diameter',
        '_outside_bead',
        '_inside_bead',
        '_x_local',
        '_y_local',
        '_z_local',
        '_r',
        '_xyz_shape'
    )

    def __init__(self, x, y, z, bead_diameter, bead_center=(0,0,0), grid=True):
        """Set up local coordinate system around bead"""
        
        self._bead_diameter = bead_diameter
        # Set up a meshgrid, or take the list of x, y and z points as 
        # coordinates
        if grid:
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        else:
            assert x.size == y.size == z.size,\
                'x, y and z need to be of the same length'
            X = np.atleast_3d(x)
            Y = np.atleast_3d(y)
            Z = np.atleast_3d(z)

        self._xyz_shape = X.shape # Store for rearranging the coordinates to original format

        # Local coordinate system around the bead and make it a vector
        self._x_local = X.reshape((1, -1)) - bead_center[0]
        self._y_local = Y.reshape((1, -1)) - bead_center[1]
        self._z_local = Z.reshape((1, -1)) - bead_center[2]

        # Calculate the distance of a coordinate to the center of the bead
        self._r = np.hypot(np.hypot(self._x_local, self._y_local), self._z_local)
        self._outside_bead = self._r > bead_diameter / 2
        self._inside_bead = self._r <= bead_diameter / 2
        

    @property
    def r(self):
        return self._r

    @property
    def r_inside(self):
        return self._r[self._inside_bead]
    
    @property
    def r_outside(self):
        return self._r[self._outside_bead]
    
    @property
    def x_inside(self):
        return self._x_local[self._inside_bead]
    
    @property
    def y_inside(self):
        return self._y_local[self._inside_bead]

    @property
    def z_inside(self):
        return self._z_local[self._inside_bead]

    @property
    def x_outside(self):
        return self._x_local[self._outside_bead]
    
    @property
    def y_outside(self):
        return self._y_local[self._outside_bead]
    
    @property
    def z_outside(self):
        return self._z_local[self._outside_bead]
       
    @property
    def region_inside_bead(self):
        return self._inside_bead
    
    @property
    def region_outside_bead(self):
        return self._outside_bead
    
    @property
    def coordinate_shape(self):
        return self._xyz_shape
    
    def xyz_stacked(self, inside:bool = False):
        if inside:
            return np.vstack((self.x_inside, self.y_inside, self.z_inside))
        else:
            return np.vstack((self.x_outside, self.y_outside, self.z_outside))

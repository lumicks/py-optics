import numpy as np
from numba import njit

@njit(cache=True)
def _exec_lookup(source, lookup, r, c):
    """
    Compile the retrieval of the lookup table with Numba, as it's slightly
    faster
    """
    return source[:, lookup[r, c]]


class DataLookup:
    """A small class that keeps data and its inverse lookup table together"""
    __slots__ = (
        "_data",
        "_inverse",
    )

    def __init__(
        self,
        data: np.ndarray,
        inverse: np.ndarray,
    ) -> None:
        self._data = data
        self._inverse = inverse

    def __call__(self, r: int, c: int):
        # A closure would have been nice, but does not lend itself to optimization by Numba as much
        # as a class with external function, in my limited benchmarking
        return _exec_lookup(self._data, self._inverse, r, c)

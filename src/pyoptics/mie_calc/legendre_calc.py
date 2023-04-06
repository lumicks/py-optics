from .legendre import *
import numpy as np
from numba import njit, prange
from joblib import Memory

cachedir = 'miecalc_cache'
memory = Memory(cachedir, verbose=0)

class AssociatedLegendreData:
    __slots__ = (
        '_associated_legendre',
        '_associated_legendre_over_sin_theta',
        '_associated_legendre_dtheta',
        '_inverse'
    )
    def __init__(
        self,
        associated_legendre: np.ndarray,
        associated_legendre_over_sin_theta: np.ndarray,
        associated_legendre_dtheta: np.ndarray,
        inverse: np.ndarray
    ) -> None:
            
        self._associated_legendre = associated_legendre
        self._associated_legendre_over_sin_theta = associated_legendre_over_sin_theta
        self._associated_legendre_dtheta = associated_legendre_dtheta
        self._inverse = inverse
    
    def associated_legendre(self, p: int, m: int):
        return self._associated_legendre[:, self._inverse[p, m]]
    
    def associated_legendre_over_sin_theta(self, p: int, m: int):
        return self._associated_legendre_over_sin_theta[:, self._inverse[p, m]]

    def associated_legendre_dtheta(self, p: int, m: int):
        return self._associated_legendre_dtheta[:, self._inverse[p, m]]

@memory.cache
def calculate_legendre(
    local_coords: np.ndarray, radii: np.ndarray, aperture: np.ndarray, 
    cos_theta: np.ndarray, sin_theta: np.ndarray,
    cos_phi: np.ndarray, sin_phi: np.ndarray,
    n_orders: int
):
    """Calculating the Legendre polynomials is computationally intense, so
    Loop over all cos(theta), in order to find the unique values of
    cos(theta)"""

    cosTs = np.zeros((*aperture.shape, radii.size))
    if (n_pupil_samples := aperture.shape[0]) != aperture.shape[1]:
        raise ValueError("Aperture matrix must be square")
    
    @njit(cache=True, parallel=True)
    def iterate(cosTs):
        for m in prange(n_pupil_samples):
            for p in range(n_pupil_samples):
                if not aperture[p, m]:
                    continue
                ct = cosTs[p, m, :]
                # Rotate the coordinate system such that the x-polarization on the
                # bead coincides with theta polarization in global space
                # however, cos(theta) is the same for phi polarization!
                # A = (_R_th(cos_theta[p,m], sin_theta[p,m]) @ 
                #     _R_phi(cos_phi[p,m], -sin_phi[p,m]))
                # coords = A @ local_coords
                z = (
                    local_coords[2, :] * cos_theta[p, m] - 
                    local_coords[0, :] * cos_phi[p, m] * sin_theta[p, m] +
                    local_coords[1, :] * sin_phi[p, m] * sin_theta[p, m]
                )
                # z = coords[2, :]
                # Retrieve an array of all values of cos(theta)
                index = np.flatnonzero(radii)
                ct[index] = z[index] / radii[index] # cos(theta)    
                ct[radii == 0] = 1
    
    iterate(cosTs=cosTs)
    cosTs = np.reshape(cosTs, cosTs.size)
    # rounding errors may make cos(theta) > 1 or < -1. Fix it to [-1..1]
    cosTs[cosTs > 1] = 1
    cosTs[cosTs < -1] = -1

    # Get the unique values of cos(theta) in the array
    cosT_unique, inverse = np.unique(cosTs, return_inverse=True)
    inverse = np.reshape(inverse, (n_pupil_samples, n_pupil_samples, radii.size))
    alp = np.zeros((n_orders, cosT_unique.size))
    alp_sin = np.zeros(alp.shape)
    alp_deriv = np.zeros(alp.shape)
    alp_prev = None # unique situation that for n == 1,
                    # the previous Assoc. Legendre Poly. isn't required
    sign_cosT_unique = np.sign(cosT_unique)
    abs_cosT_unique, sign_inv = np.unique(np.abs(cosT_unique), 
        return_inverse=True)
    parity = 1
    for L in range(1, n_orders + 1):
        alp[L - 1,:] = associated_legendre(L, abs_cosT_unique)[sign_inv]
        if parity == -1:
            alp[L - 1, :] *= sign_cosT_unique
        alp_sin[L - 1, :] = associated_legendre_over_sin_theta(
            L, cosT_unique, alp[L - 1, :]
        )
        alp_deriv[L - 1, :] = associated_legendre_dtheta(
            L, cosT_unique, (alp[L - 1, :], alp_prev)
        )
        alp_prev = alp[L - 1, :]
        parity *= -1

    return AssociatedLegendreData(alp, alp_sin, alp_deriv, inverse)


def _R_phi(cos_phi, sin_phi):
    return np.asarray([[cos_phi, -sin_phi, 0],
                        [sin_phi, cos_phi, 0],
                        [0, 0, 1]])


def _R_th(cos_th, sin_th):
    return np.asarray([[cos_th, 0, sin_th],
                        [0,  1,  0],
                        [-sin_th, 0, cos_th]])

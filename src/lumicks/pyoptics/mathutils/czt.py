"""Chirp Z transforms"""

from scipy.fft import fft, ifft, next_fast_len
import numpy as np


def init_czt(x, M, w, a):
    """Initialize auxilliary vectors that can be precomputed in order to perform a chirp-z
    transform. Storing these vectors prevents having to recalculate them on every function call,
    therefore speeding up the calculation if a transform is required on data of the same size as x
    repeatedly.

    Parameters
    ----------
    x : np.ndarray
        Vector or matrix of data. If x is a matrix, the chirp z transform is calculated along the
        columns, compatible with MATLAB's implementation. Note that the data is not actually
        transformed, but the dimensions are used to initialize the auxilliary vectors.
    M : int
        Number of points to compute the chirp-z transform at
    w : complex
        Ratio between consecutive points on the complex plane.
    a : complex
        The starting point on the complex plane of the transform.

    Returns
    -------
    precomputed : tuple
        An opaque tuple containing the auxilliary vectors.
    """
    # Upconvert to make it a matrix to handle vectors and matrices the same way
    N = x.shape[0]
    L = next_fast_len(M + N - 1)

    k = np.arange(max(M, N))
    ww = w ** (k**2 / 2)
    an = a ** -k[:N]
    anww = an * ww[:N]

    v = np.zeros(L, dtype="complex128")
    v[:M] = 1 / ww[:M]
    v[(L - N + 1) : L + 1] = 1 / ww[1:N][::-1]
    V = fft(v, L, axis=0)

    return M, L, anww, V, ww


def exec_czt(x, precomputed):
    """Calculate the chirp z transform of the discrete series x, at discrete points
    defined by init_czt() and using the precomputed auxilliary vectors for performance. This
    function performs nearly no checks on the input data.

    Tweaked implementation following [1]_.

    Parameters
    ----------
    x : np.ndarray
        Vector or matrix of data. If x is a matrix, the chirp z transform is calculated along the
        columns, minimicking MATLAB's implementation.
    precomputed : tuple
        An opaque tuple with the necessary data to perform the transform, obtained with `init_czt()`

    Returns
    -------
    X : np.ndarray
        The M-point chirp z transform of X


    ..  [1] Lawrence R. Rabiner, Ronald W. Schafer, and Charles M. Rader, "The chirp z-transform
            algorithm and its application," Bell Syst. Tech. J. 48, 1249-1292 (1969).
    """
    M, L, anww, V, ww = precomputed

    y = (anww * x.T).T
    Y = fft(y, L, axis=0)

    G = (Y.T * V).T

    g = ifft(G, L, axis=0)
    g = (g[:M].T * ww[:M]).T

    return g


def czt(x, M, w, a):
    """Calculate the chirp z transform of the discrete series x, at discrete points
    on the complex plane defined by z = a * w**(-(0:M-1)).

    Direct implementation following [1]_.

    Parameters
    ----------
    x : np.ndarray
        Vector or matrix of data. If x is a matrix, the chirp z transform is calculated along the
        columns, compatible with MATLAB's implementation.
    M : int
        Number of points to calculate the transform for.
    w : complex
        Ratio between consecutive points on the complex plane.
    a : complex
        The starting point on the complex plane of the transform.

    Returns
    -------
    X : np.ndarray
        The M-point chirp z transform.


    ..  [1] Lawrence R. Rabiner, Ronald W. Schafer, and Charles M. Rader, "The chirp z-transform
            algorithm and its application," Bell Syst. Tech. J. 48, 1249-1292 (1969).
    """

    N = x.shape[0]
    L = next_fast_len(M + N - 1)

    k = np.arange(max(M, N))
    ww = w ** (k**2 / 2)
    an = a ** -k[:N]
    anww = an * ww[:N]
    y = (anww * x.T).T
    Y = fft(y, L, axis=0)

    v = np.zeros(L, dtype="complex128")
    v[:M] = 1 / ww[:M]
    v[(L - N + 1) : L + 1] = 1 / ww[1:N][::-1]
    V = fft(v, L, axis=0)
    G = (Y.T * V).T

    g = ifft(G, L, axis=0)
    g = (g[:M].T * ww[:M]).T

    return g

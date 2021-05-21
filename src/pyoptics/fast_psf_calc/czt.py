"""Chirp Z transforms"""

from mkl_fft._numpy_fft import fft, ifft
import numpy as np

def init_czt(x, M, w, a):
    """Initialize auxilliary vectors that can be precomputed in order to perform a chirp-z
    transform. Storing these vectors prevents having to recalculate them on every function
    call, therefore speeding up the calculation if a transform is required on data of the
    same size as x repeatedly.

    Args:
        x: vector or matrix of data. If x is a matrix, the chirp z transform is calculated
            along the columns, compatible with MATLAB's implementation. Note that the data is
            not actually transformed, but the dimensions are used to initialize the
            auxilliary vectors.
        M: Number of points to compute the chirp-z transform at
        w: ratio between consecutive points on the complex plane.
        a: The starting point on the complex plane of the transform.

    Returns:
        An opaque tuple containing the auxilliary vectors.
    """
     # Upconvert to make it a matrix to handle vectors and matrices the same way
    xshape = x.shape
    x = np.atleast_2d(x)

    newshape = x.shape
    N = int(newshape[0])
    expand_shape = (1, *newshape[1:])
    tile_shape = [1] * (len(newshape)-1)

    L =  1 << ((M + N - 1) - 1).bit_length()  # pyfftw.next_fast_len(M + N  -1)

    k = np.arange(np.max((M, N))).T
    ww = w**(k**2 / 2)
    an = a**-(k[:N])
    anww = an * ww[0:N]
    anww = anww.reshape((N, *tile_shape))
    anww = np.tile(anww, expand_shape)

    v = np.zeros(L, dtype='complex128')
    v[0: M] = 1 / ww[0: M]
    v[(L - N + 1):L+1] = 1 / ww[1:N][::-1]

    V = fft(v, L, axis=0)
    V = np.reshape(V, (L, *tile_shape))
    V = np.tile(V, expand_shape)

    ww = ww.reshape((ww.shape[0], *tile_shape))
    ww = np.tile(ww[0:M], expand_shape)

    return xshape, M, L, anww, V, ww


def exec_czt(x, precomputed):
    """Calculate the chirp z transform of the discrete series x, at discrete points
    defined by init_czt() and using the precomputed auxilliary vectors for performance.
    This function performs nearly no checks on the input data.

    Tweaked implementation following Lawrence R. Rabiner, Ronald W. Schafer,
    and Charles M. Rader, "The chirp z-transform algorithm and its
    application," Bell Syst. Tech. J. 48, 1249-1292 (1969).

    Args:
        x: vector or matrix of data. If x is a matrix, the chirp z transform is calculated
            along the columns, minimicking MATLAB's implementation.
        precomputed: an opaque tuple with the necessary data to perform the transform,
            obtained with init_czt()

    Returns:
        The M-point chirp z transform
    """
     # Upconvert to make it a matrix to handle vectors and matrices the same way
    x=np.atleast_2d(x)
    xshape, M, L, anww, V, ww = precomputed

    y = anww * x
    Y = fft(y, L, axis=0)

    G = Y * V

    g = ifft(G, L, axis=0)
    g = g[0:M] * ww

    if len(xshape) == 1:
        output = g.reshape((M,))
    else:
        output = g

    return output


def czt(x, M, w, a):
    """Calculate the chirp z transform of the discrete series x, at discrete points
    on the complex plane defined by z = a * w**(-(0:M-1)).

    Direct implementation following Lawrence R. Rabiner, Ronald W. Schafer,
    and Charles M. Rader, "The chirp z-transform algorithm and its
    application," Bell Syst. Tech. J. 48, 1249-1292 (1969).

    Args:
        x: vector or matrix of data. If x is a matrix, the chirp z transform is calculated
            along the columns, compatible with MATLAB's implementation.
        M: number of points to calculate the transform for.
        w: ratio between consecutive points on the complex plane.
        a: The starting point on the complex plane of the transform.

    Returns:
        The M-point chirp z transform.
    """

    # Upconvert to make it a matrix to handle vectors and matrices the same way
    xshape=x.shape
    x=np.atleast_2d(x)

    newshape = x.shape
    N = int(newshape[0])
    expand_shape = (1, *newshape[1:])
    tile_shape = [1] * (len(newshape)-1)

    L = 1 << ((M + N - 1) - 1).bit_length()

    k = np.arange(np.max((M, N))).T
    ww = w**(k**2 / 2)
    an = a**-(k[:N])
    anww = an * ww[0:N]
    anww = anww.reshape((N, *tile_shape))
    y = np.tile(anww, expand_shape) * x
    Y = fft(y, L, axis=0)

    v = np.zeros(L, dtype='complex128')
    v[0: M] = 1 / ww[0: M]
    v[(L - N + 1):L+1] = 1 / ww[1:N][::-1]

    V = fft(v, L, axis=0)
    V = np.reshape(V, (L, *tile_shape))
    G = Y * np.tile(V, expand_shape)

    g = ifft(G, L, axis=0)
    ww = ww.reshape((ww.shape[0], *tile_shape))
    g = g[0:M] * np.tile(ww[0:M], expand_shape)

    if len(xshape) == 1:
        output = g.reshape((M,))
    else:
        output = g

    return output

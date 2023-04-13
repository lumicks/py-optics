# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[]
import numpy as np
import matplotlib.pyplot as plt
from pyoptics import mie_calc as mc
import pyinstrument
import mpmath as mp

# + tags=[]
order = 100
x = np.linspace(0., 1, 1000000)
profiler = pyinstrument.Profiler()
try:
    profiler.start()
    for order in range(30):
        alp = mc.associated_legendre(order, x)
        _ = mc.associated_legendre_dtheta(order, x, (alp, alp))
        _ = mc.associated_legendre_over_sin_theta(order, x, alp)
finally:
    profiler.stop()
profiler.open_in_browser()


# + tags=[]
def associated_legendre_decimal(n: int, x: mp.mpf):
    """associated_legendre(n, x): Return the 1st order (m == 1) of the 
    associated Legendre polynomial of degree n, evaluated at x [-1..1]

    Uses a Clenshaw recursive algorithm for the evaluation, specific for the 
    1st order associated Legendre polynomials.

    See Clenshaw, C. W. (July 1955). "A note on the summation of Chebyshev 
    series". Mathematical Tables and Other Aids to Computation. 9 (51): 118
    """
    def _fi1(x):
        """First order associated Legendre polynomial evaluated at x"""
        return -((1 + x) * (1 - x))**mp.mpf('0.5')

    def _fi2(x):
        """Second order associated Legendre polynomial evaluated at x"""
        return -3 * x * ((1 + x) * (1 - x))**mp.mpf('0.5')

    if n == 1:
        return _fi1(x)
    if n == 2:
        return _fi2(x)
    bk2 = mp.mpf('0')
    bk1 = mp.mpf('1')
    for k_ in range(n - 1, 1, -1):
        k = mp.mpf(k_)
        bk = ((2 * k + 1) / k) * x * bk1 - (k + 2) / (k + 1) * bk2
        bk2 = bk1
        bk1 = bk
    return _fi2(x) * bk1 - mp.mpf('1.5') * _fi1(x) * bk2


# + tags=[]
mp.mp.dps = 45

alp_dm = []
order = 100

for _x in x:
    alp_dm.append(associated_legendre_decimal(order, _x))

# + tags=[]
error_fm = []
for fm, _dm in zip(alp_fm, alp_dm):
    error_fm.append(float(fm -_dm))

# + tags=[]
error = []
for fm, al in zip(alp, alp_dm):
    error.append(float(mp.mpf(fm)-al))


# + tags=[]
# print(type(error[0]))
plt.figure(figsize=(8, 12))
plt.hist(error, 50)
plt.hist(error_fm, 50)
plt.yscale('log')
# plt.plot(x, alp_fm)
plt.show()

# + tags=[]
plt.figure(figsize=(8, 12))
plt.plot(x, alp)
plt.show()
# -



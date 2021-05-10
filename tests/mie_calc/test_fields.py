# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3.8 (XPython)
#     language: python
#     name: xpython
# ---

import numpy as np
import numpy.testing
import pytest

import pyoptics.mie_calc as mc

# +
beadsize = 4e-6
x = np.linspace(-beadsize, beadsize, 100)
y = 0
z = x
mie = mc.MieCalc(beadsize, 1.5, 1.0, 1064e-9)

Ex, Ey, Ez, X, Y, Z = mie.fields_plane_wave(x=x, y=y, z=z, theta=0, phi=0,
                                                        polarization=(0,1), return_grid=True, verbose=False, total_field=False)
# -

# %matplotlib inline
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.pcolor(Z, X, Ex.real, cmap='jet', shading='auto')
plt.show()



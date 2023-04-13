# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.8 (XPython)
#     language: python
#     name: xpython
# ---

# %%
import numpy as np
import numpy.testing
import pytest

# %% tags=[]
import pyoptics.mie_calc as mc


# %%
@pytest.mark.parametrize("n_medium", [1.0, 1.33, 1.5])
@pytest.mark.parametrize("n_bead", [1.0, 1.33, 1.5, 2.5])
def test_quasi_static_field(
    n_medium, n_bead
):
    # Electric field in a small dielectric particle (<< lambda) can be approximated with quasistatic solution:
    E = 3/(2 + n_bead**2/n_medium**2)

    bead = mc.Bead(1e-3, n_bead, n_medium, 1)
    Ex, Ey, Ez = mc.fields_plane_wave(bead, x=0, y=0, z=0, theta=0, phi=0, polarization=(1,0))

    np.testing.assert_allclose([Ex, Ey, Ez], [E, 0, 0], rtol=1e-4, atol=1e-14)
    Ex, Ey, Ez = mc.fields_plane_wave(bead, x=0, y=0, z=0, theta=0, phi=0, polarization=(0,1))
    
    np.testing.assert_allclose([Ex, Ey, Ez], [0, E, 0], rtol=1e-4, atol=1e-14)



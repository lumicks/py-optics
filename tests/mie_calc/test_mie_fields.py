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
import os

import pyoptics.mie_calc as mc


# + [markdown] tags=[]
# Reference data sets are calculated by miepy 0.5.0, by the script `generate_benchmark_fields.py`
# -

@pytest.mark.parametrize('dataset',[1,2,3,4])
def test_mie_nearfield(dataset):
    path = os.path.abspath(os.path.dirname(__file__))

    data = np.load(os.path.join(path, f'ref{dataset}.npz'))
    try:
        Exr = data['Ex']
        Eyr = data['Ey']
        Ezr = data['Ez']
        x = data['x']
        y = data['y']
        z = data['z']
        n_bead = data['n_bead']
        n_medium = data['n_medium']
        bead_diam = data['bead_diam']
        lambda_vac = data['lambda_vac']
        num_pts = data['num_pts']
        num_orders = data['num_orders']
    finally:
        data.close()

    mie = mc.MieCalc(bead_diam, n_bead, n_medium, lambda_vac)
    Ex, Ey, Ez, X, Y, Z = mie.fields_plane_wave(x, y, z, num_orders=num_orders, return_grid=True, inside_bead=True, total_field=False)
    err_x = Ex - Exr
    err_y = Ey - Eyr
    err_z = Ez - Ezr

    np.testing.assert_allclose(Ex, Exr, rtol=1e-3)
    np.testing.assert_allclose(Ey, Eyr, rtol=1e-3)
    np.testing.assert_allclose(Ez, Ezr, rtol=1e-3)


@pytest.mark.parametrize('bead_diameter, n_bead',[(4e-6, 1.5),(0.2e-6, 0.1 + 2j)])
def test_mie_nearfield_polarizations(bead_diameter, n_bead):
    
    x = np.linspace(-bead_diameter, bead_diameter, 99)
    mie = mc.MieCalc(bead_diameter, n_bead, n_medium=1.33, lambda_vac=1064e-9)
    Ext, Eyt, Ezt = mie.fields_plane_wave(x=x, y=x, z=x, return_grid=False, inside_bead=True, total_field=True)
    Exp, Eyp, Ezp, X, Y, Z = mie.fields_plane_wave(x=x, y=x, z=x, polarization=(0,1), return_grid=True, inside_bead=True, total_field=True)
    
    np.testing.assert_allclose(Ext, np.rot90(Eyp), rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(Eyt, Exp, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(np.rot90(Ezt), Ezp, rtol=1e-8, atol=1e-14)

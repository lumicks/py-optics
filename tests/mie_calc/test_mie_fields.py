import numpy as np
import pytest
import os
from scipy.constants import (
    speed_of_light as C,
    mu_0 as MU0,
)

import pyoptics.mie_calc as mc

data_location = 'test_data'
data_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(data_path, data_location)

data_available = True
for dataset in (1, 2, 3, 4, 5):
    data_available &= os.path.exists(os.path.join(data_path, f'ref{dataset}.npz'))

# Reference data sets are calculated by miepy 0.5.0, by the script
# `generate_benchmark_fields.py`
@pytest.mark.skipif(not data_available, 
                    reason="skip fields test, data not present")
@pytest.mark.parametrize('dataset', [1, 2, 3, 4, 5])
def test_mie_nearfield(dataset):
    data = np.load(os.path.join(data_location, f'ref{dataset}.npz'))
    try:
        Exr = data['Ex']
        Eyr = data['Ey']
        Ezr = data['Ez']
        Hxr = data['Hx']
        Hyr = data['Hy']
        Hzr = data['Hz']
        x = data['x']
        y = data['y']
        z = data['z']
        n_bead = data['n_bead']
        n_medium = data['n_medium']
        bead_diam = data['bead_diam']
        lambda_vac = data['lambda_vac']
        num_orders = data['num_orders']
    finally:
        data.close()

    bead = mc.Bead(bead_diam, n_bead, n_medium, lambda_vac)
    Ex, Ey, Ez, Hx, Hy, Hz = mc.fields_plane_wave(
        bead, x, y, z, num_orders=num_orders, return_grid=False,
        magnetic_field=True, total_field=False
    )

    np.testing.assert_allclose(Ex, Exr, rtol=1e-3)
    np.testing.assert_allclose(Ey, Eyr, rtol=1e-3)
    np.testing.assert_allclose(Ez, Ezr, rtol=1e-3)
    np.testing.assert_allclose(Hx*C*MU0, Hxr, rtol=1e-3)
    np.testing.assert_allclose(Hy*C*MU0, Hyr, rtol=1e-3)
    np.testing.assert_allclose(Hz*C*MU0, Hzr, rtol=1e-3)


@pytest.mark.parametrize(
    'bead_diameter, n_bead', [(4e-6, 1.5), (0.2e-6, 0.1 + 2j)]
)
def test_mie_nearfield_polarizations(bead_diameter, n_bead):

    x = np.linspace(-bead_diameter, bead_diameter, 99)
    bead = mc.Bead(bead_diameter, n_bead, n_medium=1.33, lambda_vac=1064e-9)
    Ext, Eyt, Ezt = mc.fields_plane_wave(
        bead, x=x, y=x, z=x, return_grid=False, total_field=True
    )
    Exp, Eyp, Ezp = mc.fields_plane_wave(
        bead, x=x, y=x, z=x, polarization=(0, 1), return_grid=False,
        total_field=True
    )

    np.testing.assert_allclose(Ext, np.rot90(Eyp), rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(Eyt, Exp, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(np.rot90(Ezt), Ezp, rtol=1e-8, atol=1e-14)

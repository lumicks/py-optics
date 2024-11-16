# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bluelake splash screen
# Code to calculate the image displayed on Bluelake 2.5.
# *Note:* this code probably needs a computer with at least 16 GB of memory.

# %%
import matplotlib.pyplot as plt

# %matplotlib inline
import numpy as np

from lumicks.pyoptics import trapping as trp

# %%
bead_diameter = 4.05e-6  # [m]
lambda_vac = 1.064e-6  # [m]
n_bead = 1.5  # [-]
n_medium = 1.33  # [-]
bead = trp.Bead(
    bead_diameter=bead_diameter, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac
)
focal_length = 4.43e-3  # [m]
NA = 0.9  # [-]
n_bfp = 1.0  # [-]
objective = trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)
bfp_sampling_n = 11  # This is relatively low, but good enough for an illustration

# %%
num_pts_z = 992
num_pts_x = 360  # Target is an image of 992 x 512 pixels, but we'll use symmetry
dz = 13e-6 / num_pts_z
z = (np.arange(num_pts_z) - num_pts_z / 2) * dz
x = np.arange(num_pts_x) * dz
bead_center = (0, 0, 317.6e-9)  # [m], equilibrium position from another simulation
Ex, Ey, Ez, X, Y, Z = trp.fields_focus_gaussian(
    1,
    filling_factor=0.9,
    objective=objective,
    bead=bead,
    x=x,
    y=0,
    z=z,
    num_orders=12,  # lower number of orders such that the image looks good, but we need less memory
    bead_center=bead_center,
    bfp_sampling_n=bfp_sampling_n,
    return_grid=True,
    verbose=True,
)

# %%
_E = (np.abs(Ez) ** 2 + np.abs(Ex) ** 2) ** 0.5  # field strength [V/m]
E = np.empty((512, 992))
E[512 - 360 : 512, :] = _E
E[0 : 512 - 360, :] = np.flipud(_E[0 : 512 - 360, :])
E = np.flipud(E)
plt.figure(figsize=(8, 6))
plt.imshow(E, cmap="inferno")
plt.show()

# %%
plt.imsave("splash.png", E, cmap="inferno")

# %%
num_pts_z = 992
num_pts_x = 64  # Target is an image of 992 x 128 pixels, but we'll use symmetry
dz = 13e-6 / num_pts_z
z = (np.arange(num_pts_z) - num_pts_z / 2) * dz
x = np.arange(num_pts_x) * dz
bead_center = (0, 0, 317.6e-9)  # [m], equilibrium position from another simulation
Ex, Ey, Ez, X, Y, Z = trp.fields_focus_gaussian(
    1,
    filling_factor=0.9,
    objective=objective,
    bead=bead,
    x=x,
    y=0,
    z=z,
    num_orders=12,  # lower the number of orders such that the image looks good, but we need less memory
    bead_center=bead_center,
    bfp_sampling_n=bfp_sampling_n,
    return_grid=True,
    verbose=True,
)

# %%
_E = (np.abs(Ez) ** 2 + np.abs(Ex) ** 2) ** 0.5  # field strength [V/m]
E = np.empty((128, 992))
E[128 - 64 : 128, :] = _E
E[0 : 128 - 64, :] = np.flipud(_E[0 : 128 - 64, :])
E = np.flipud(E)
plt.figure(figsize=(8, 6))
plt.imshow(E, cmap="inferno")
plt.show()

# %%
plt.imsave("header.png", E, cmap="inferno")

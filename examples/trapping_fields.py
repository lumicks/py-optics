# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calculating field around a bead

# %%
import matplotlib.pyplot as plt

# %matplotlib inline
import numpy as np

from lumicks.pyoptics import trapping as trp

# %% [markdown]
# ## Definition of coordinate system
# The optical axis (direction of the light to travel into) is the $+z$ axis. For an aberration-free system, the focus of the laser beam ends up at $(x, y, z) = (0, 0, 0)$ See also below:
#
# <div>
#     <img src="images/axes.png" width=400>
# </div>
#
# ## Properties of the bead, the medium and the laser
# The bead is described by a refractive index $n_{bead}$, a diameter $D$ and a location in space $(x_b, y_b, z_b)$, the latter two in meters. In the code, the diameter is given by `bead_diameter`. The refractive index is given by `n_bead` and the location is passed to the code as a tuple `bead_center` containing three floating point numbers. These numbers represent the $x$-, $y$- and $z$-location of the bead, respectively, in meters. The wavelength of the trapping light is given in meters as well, by the parameter `lambda_vac`. The wavelength is given as it occurs in vacuum ('air'), not in the medium. The refractive index of the medium $n_{medium}$ is given by the parameter `n_medium`.
#
# ### 1 micron polystyrene particle in water
# Laser at 1064 nm

# %%
bead_diameter = 1.0e-6  # [m]
lambda_vac = 1.064e-6  # [m]
n_bead = 1.57  # [-]
n_medium = 1.33  # [-]
bead = trp.Bead(
    bead_diameter=bead_diameter, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac
)

# %% [markdown]
# ## Properties of the objective
# See the image below. The definition of the coordinate system remains as before, and the objective is described by the $\mathit{NA}=n_\mathit{medium} \sin(\theta)$, the focal length $f$ in meters, and the medium at the back focal plane (BFP), $n_\mathit{bfp}$. The parameter `NA` sets the $\mathit{NA}$ (unitless), `focal_length` sets the focal length $f$ (in meters) and the refractive index of the medium at the BFP, $n_\mathit{bfp}$, is set by `n_bfp` (unitless).
# <div>
#     <img src="images/objective.png" width=400>
# </div>

# %%
# This describes a water-immersion objective used for trapping quite well:
focal_length = 4.43e-3  # [m]
NA = 1.2  # [-]
n_bfp = 1.0  # [-]
objective = trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

# %% [markdown]
# ## Calculate the fields around the bead
# Set up the area to calculate around the focus (coordinates (0, 0, 0)), the number of points to calculate, and where the bead is

# %%
# Number of points, area to show:
num_pts = 201
extent = 2 * bead_diameter
x = np.arange(num_pts) * extent / (num_pts - 1) - extent / 2
z = x

# Bead location relative to the focus:
bead_center = (0, 0, 0)  # [m]

# Calculate:
Ex, Ey, Ez, X, Y, Z = trp.fields_focus_gaussian(
    1,
    filling_factor=0.9,
    objective=objective,
    bead=bead,
    x=x,
    y=0,
    z=z,
    bead_center=bead_center,
    return_grid=True,
    verbose=True,
)

# %%
plt.figure(figsize=(10, 8))
plt.pcolor(
    Z * 1e6, X * 1e6, np.sqrt(np.abs(Ez) ** 2 + np.abs(Ex) ** 2), cmap="plasma", shading="auto"
)
plt.xlabel("Z [µm]")
plt.ylabel("X [µm]")
plt.title("|E| [V/m]")
plt.colorbar()
plt.show()

# %% [markdown]
# Imaginary part of the z-component of the E field:

# %%
plt.figure(figsize=(10, 8))
plt.pcolor(Z * 1e6, X * 1e6, Ez.imag, cmap="plasma", shading="auto")
plt.colorbar()
plt.title("imag(Ez) [V/m]")
plt.xlabel("Z [µm]")
plt.ylabel("X [µm]")
plt.show()

# %% [markdown]
# ## 4.05 mu bead 'kissing' the focus
# Shows bead off-center from focus, bead in air, NA of 0.9. Also, a cross section over the x-plane is taken. Because of symmetry, Ez and Ey should be zero in this plane.

# %%
bead_size = 4.05e-6
lambda_vac = 1.064e-6
n_bead = 1.5
n_medium = 1.0
num_pts = 101
y = np.linspace(-0.5 * bead_size, 1.5 * bead_size, num_pts)
z = np.linspace(-1 * bead_size, 1 * bead_size, num_pts)
bead = trp.Bead(bead_diameter=bead_size, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac)
objective = trp.Objective(NA=0.9, focal_length=4.43e-3, n_medium=n_medium, n_bfp=1.0)

# %%
Ex, Ey, Ez, X, Y, Z = trp.fields_focus_gaussian(
    1,
    filling_factor=0.9,
    objective=objective,
    bead=bead,
    x=0,
    y=y,
    z=z,
    bead_center=(0, 2.9e-6, 0),
    return_grid=True,
    verbose=True,
)

# %%
plt.figure(figsize=(10, 8))
plt.pcolor(Z * 1e6, Y * 1e6, np.abs(Ex), cmap="plasma", shading="auto")
plt.colorbar()
plt.title("|E| [V/m]")
plt.xlabel("Z [µm]")
plt.ylabel("Y [µm]")
plt.show()

# %% [markdown]
# Imaginary part of Ex. Modify the code to check that Ez and Ey are practically zero too:

# %%
plt.figure(figsize=(10, 8))
plt.pcolor(Z * 1e6, Y * 1e6, Ex.imag, cmap="plasma", shading="auto")
plt.xlabel("Z [µm]")
plt.ylabel("Y [µm]")
plt.title("imag(Ex) [V/m]")
plt.colorbar()
plt.show()

# %% [markdown]
# ## Silver nanoparticle, excited by 345 nm UV laser and 1064 nm IR laser

# %%
bead_diameter = 0.15e-6  # [m]
lambda_vac_UV = 0.345e-6  # [m]
n_bead_UV = 0.06 + 1.76j  # [-] Silver at 345 nm
lambda_vac_IR = 1064e-9  # [m]
n_bead_IR = 0.04 + 7.60j  # [-] Silver at 1064 nm
n_medium = 1.0
num_pts = 101
half_extent = bead_diameter
# np.linspace can return asymmetric values for negative and positive sides of an axis.
# The code below ensures that the negative values of x are the same
# as the positive values of x, except for the sign. This helps to reduce the number
# of points to calculate by taking advantage of symmetry.
x = np.zeros(num_pts * 2 - 1)
_x = np.linspace(0, half_extent, num_pts)
x[0:num_pts] = -_x[::-1]
x[num_pts:] = _x[1:]
z = x

bead_UV = trp.Bead(
    bead_diameter=bead_diameter, n_bead=n_bead_UV, n_medium=n_medium, lambda_vac=lambda_vac_UV
)
bead_IR = trp.Bead(
    bead_diameter=bead_diameter, n_bead=n_bead_IR, n_medium=n_medium, lambda_vac=lambda_vac_IR
)
objective = trp.Objective(NA=0.9, focal_length=4.43e-3, n_medium=n_medium, n_bfp=1.0)

# %%
Ex_UV, Ey_UV, Ez_UV, X, Y, Z = trp.fields_focus_gaussian(
    1,
    filling_factor=0.9,
    objective=objective,
    bead=bead_UV,
    x=x,
    y=0,
    z=z,
    bead_center=(0, 0, 0),
    return_grid=True,
    verbose=True,
)

# %%
Ex_IR, Ey_IR, Ez_IR, X, Y, Z = trp.fields_focus_gaussian(
    1,
    filling_factor=0.9,
    objective=objective,
    bead=bead_IR,
    x=x,
    y=0,
    z=z,
    bead_center=(0, 0, 0),
    return_grid=True,
    verbose=True,
)

# %%
plt.figure(figsize=(21, 7.5))
plt.subplot(1, 2, 1)
plt.pcolormesh(
    Z, X, np.sqrt(np.abs(Ez_IR) ** 2 + np.abs(Ex_IR) ** 2), cmap="plasma", shading="auto"
)
plt.colorbar()
plt.title("|$E_{IR}$|")
plt.xlabel("Z [µm]")
plt.ylabel("X [µm]")

plt.subplot(1, 2, 2)
plt.pcolormesh(
    Z * 1e6,
    X * 1e6,
    np.sqrt(np.abs(Ez_UV) ** 2 + np.abs(Ex_UV) ** 2),
    cmap="plasma",
    shading="auto",
)
plt.colorbar()
plt.title("|$E_{UV}$|")
plt.xlabel("Z [µm]")
plt.ylabel("X [µm]")

plt.show()

# %%
plt.figure(figsize=(21, 7.5))
plt.subplot(1, 2, 1)
plt.pcolormesh(Z * 1e9, X * 1e9, Ex_IR.imag, cmap="plasma", shading="auto")
plt.colorbar()
plt.title("imag($E_{x,IR}$) [V/m]")
plt.xlabel("Z [nm]")
plt.ylabel("X [nm]")

plt.subplot(1, 2, 2)
plt.pcolormesh(Z * 1e9, X * 1e9, Ex_UV.imag, cmap="plasma", shading="auto")
plt.colorbar()
plt.title("imag($E_{x,UV}$) [V/m]")
plt.xlabel("Z [nm]")
plt.ylabel("X [nm]")

plt.show()

# %% [markdown]
# ### Showcase a few other options

# %%
bead_size = 1e-6  # [m]
lambda_vac = 1.064e-6  # [m]
n_bead = 1.57  # [-]
n_medium = 1.33  # [-]
num_pts = 501
x = np.linspace(-1 * bead_size, 1 * bead_size, num_pts)
z = np.linspace(-1 * bead_size, 1 * bead_size, num_pts)
bead = trp.Bead(bead_diameter=bead_size, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac)

# %% [markdown]
# Limit the number of orders to 1 (dipole) by using the `num_spherical_modes` argument. Note the incorrect representation of the fields.
# Change `num_spherical_modes` to `None` or remove it to restore default behavior, which is to use the recommended number of orders from literature

# %%
Ex, Ey, Ez, X, Y, Z = trp.fields_plane_wave(
    bead, x=x, y=0, z=z, num_spherical_modes=1, return_grid=True
)

# %%
plt.figure(figsize=(10, 8))
plt.pcolormesh(
    Z * 1e6, X * 1e6, np.sqrt(np.abs(Ez) ** 2 + np.abs(Ex) ** 2), cmap="plasma", shading="auto"
)
plt.colorbar()
plt.title("|E| [V/m]")
plt.xlabel("Z [µm]")
plt.ylabel("X [µm]")
plt.show()

# %% [markdown]
# Ask how many orders are necessary according to the literature, then use half

# %%
num_modes = bead.number_of_modes
print(f"Number of modes is {num_modes}")
Ex, Ey, Ez, X, Y, Z = trp.fields_plane_wave(
    bead, x=x, y=0, z=z, num_spherical_modes=np.max((num_modes // 2, 1)), return_grid=True
)

# %%
plt.figure(figsize=(10, 8))
plt.pcolormesh(
    Z * 1e6, X * 1e6, np.sqrt(np.abs(Ez) ** 2 + np.abs(Ex) ** 2), cmap="plasma", shading="auto"
)
plt.colorbar()
plt.title("|E| [V/m]")
plt.xlabel("Z [µm]")
plt.ylabel("X [µm]")
plt.show()
plt.show()

# %% [markdown]
# Get the scattering coefficients $a_n$ and $b_n$

# %%
an, bn = bead.ab_coeffs()

plt.figure(figsize=(10, 8))
plt.semilogy(range(1, an.size + 1), np.abs(an), label="$|a_n|$")
plt.semilogy(range(1, bn.size + 1), np.abs(bn), label="$|b_n|$")
plt.xlabel("Order [n]")
plt.legend()
plt.title("$|a_n|$ and $|b_n|$")
plt.show()

# %% [markdown]
# Only calculate the scattered field (so not the focused laser). Change the bead from small to larger to see the transitions from Rayleigh scattering to forward scattering

# %%
bead_size = 0.1e-6  # [m]
lambda_vac = 1.064e-6  # [m]
n_bead = 1.57  # [-]
n_medium = 1.33  # [-]
num_pts = 401
x = np.linspace(-2 * bead_size, 2 * bead_size, num_pts)
z = np.linspace(-2 * bead_size, 2 * bead_size, num_pts)
bead = trp.Bead(bead_diameter=bead_size, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac)

# %%
Ex, Ey, Ez, X, Y, Z = trp.fields_plane_wave(
    bead=bead, x=x, y=0, z=z, num_spherical_modes=None, total_field=False, return_grid=True
)

# %%
plt.figure(figsize=(10, 8))
plt.pcolormesh(
    Z * 1e6,
    X * 1e6,
    np.log10(np.sqrt(np.abs(Ez) ** 2 + np.abs(Ex) ** 2)),
    cmap="plasma",
    shading="auto",
)
plt.colorbar()
plt.title("|E| (scattered)")
plt.xlabel("Z [µm]")
plt.ylabel("X [µm]")
plt.show()

# %%

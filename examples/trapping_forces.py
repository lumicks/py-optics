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
# # Calculating forces on a trapped bead

# %%
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0
from scipy.constants import speed_of_light as C
from scipy.interpolate import interp1d

import lumicks.pyoptics.trapping as trp

# %% [markdown]
# ## Definition of coordinate system
# The optical axis (direction of the light to travel into) is the $+z$ axis. For an aberration-free system, the focus of the laser beam ends up at $(x, y, z) = (0, 0, 0)$ See also below:
#
# <figure>
#     <img src="images/axes.png" width=400>
#     <figcaption>Fig. 1: Definition of the coordinate system</figcaption>
# </figure>
#
# ## Properties of the bead, the medium and the laser
# The bead is described by a refractive index $n_{bead}$, a diameter $D$ and a location in space $(x_b, y_b, z_b)$, the latter two in meters. In the code, the diameter is given by `bead_diameter`. The refractive index is given by `n_bead` and the location is passed to the code as a tuple `bead_center` containing three floating point numbers. These numbers represent the $x$-, $y$- and $z$-location of the bead, respectively, in meters. The wavelength of the trapping light is given in meters as well, by the parameter `lambda_vac`. The wavelength is given as it occurs in vacuum ('air'), not in the medium. The refractive index of the medium $n_{medium}$ is given by the parameter `n_medium`.

# %%
bead_diameter = 4.4e-6  # [m]
lambda_vac = 1064e-9  # [m]
n_bead = 1.57  # [-]
n_medium = 1.33  # [-]

# %%
# instantiate a Bead object
bead = trp.Bead(
    bead_diameter=bead_diameter, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac
)
# Tell use how many scattering orders are used according to the formula in literature:
print(f"Number of scattering orders used by default: {bead.number_of_modes}")

# %% [markdown]
# ## Properties of the objective
# See the image below. The definition of the coordinate system remains as before, and the objective is described by the $\mathit{NA}=n_\mathit{medium} \sin(\theta)$, the focal length $f$ in meters, and the medium at the back focal plane (BFP), $n_\mathit{bfp}$. The parameter `NA` sets the $\mathit{NA}$ (unitless), `focal_length` sets the focal length $f$ (in meters) and the refractive index of the medium at the BFP, $n_\mathit{bfp}$, is set by `n_bfp` (unitless).
# <figure>
#     <img src="images/objective.png" width=400>
#     <figcaption>Fig. 2: Coordinate system and the relation to the properties of the objective (NA, focal length, immersion medium)</figcaption>
# </figure>
#

# %%
# objective properties, for water immersion
NA = 1.2  # [-]
focal_length = 4.43e-3  # [m]
n_bfp = 1.0  # [-] Other side of the water immersion objective is air
# Instantiate an Objective. Note that n_medium has to be defined here as well
objective = trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

# %% [markdown]
# ## Properties of the input beam
#
# The continuous field distribution at the back focal plane (the input beam, left figure) is numerically integrated. This numerical integration uses discrete samples. Shown here on the right is the case of equidistant sampling, with $(2N-1) \times (2N-1)$ samples, counting $N$ from the center to the extremes of the $x$ and $y$ axes (right figure, here $N=9$). The more samples, the more accurate the resulting point spread function will represent the continuous distribution. But, the calculation time will increase with $O(N^2)$. You can get away with relatively small numbers as long as the distance between the bead and the focus is not too large. *Do* check for convergence, however, for real applications (see below).
# <figure>
#     <img src="images/aperture_inf.png" width=400 align="left">
#     <img src="images/aperture_discretized.png" width=400>
#     <figcaption>Fig. 3: Left - continuous field distribution of the laser beam. Right - discretized approximation</figcaption>
# </figure>

# %% [markdown]
# # Beam type and power
#
# In order to calculate the force on a bead, the total power in the beam needs to be set, and the type or profile of beam. The beam profile is typically approximated with a Gaussian distribution, which is also the approach taken here.

# %% Set the maximum power at 1.75W for a single trapping beam, where the power is valid before any
# clipping by the objective.
Pmax = 1.75  # [W]
# Reduce the power to a more typical one used during experiments:
power_percentage = 25  # [%]

# %%
filling_factor = 0.9  # [-]
P = Pmax * power_percentage / 100.0  # [W]


# Determine the electric field distribution at the back focal plane, for a given power and filling
# factor
def gaussian_beam(coordinates, objective, *, power, filling_factor):
    w0 = filling_factor * objective.r_bfp_max  # [m]
    I0 = 2 * power / (np.pi * w0**2)  # [W/m^2]
    E0 = (I0 * 2 / (epsilon_0 * C * n_bfp)) ** 0.5  # [V/m]

    Ex = np.exp(-(coordinates.x_bfp**2 + coordinates.y_bfp**2) / w0**2) * E0
    return (Ex, None)


# %% [markdown]
# ## Forces in the $z$ direction - find the equilibrium
# Set the range in z to calculate the forces at. We expect the force on the bead in the z-direction to be zero for the interval of $[0, \inf)$
# Start with a range of $0\ldots 1 \mu m$. We will retrieve a function from `force_factory` that we can reuse to calculate the force on a bead at arbitrary locations. To set up the correct order of integration, we'll use the `Objective.minimal_integration_order()` method to help.

# %%
# Set up the coordinate system. We'll calculate the force along the optical axis at 50 nm intervals
num_pts = 201
z = np.linspace(0, 1e-6, num_pts)

# Later we'll calculate along x and y as well, and to set up the return value of
# `trapping.force_factory` to converge for these bead center locations, we'll pass those coordinates
# as well
x = np.linspace(-500e-9, 0, num_pts)
y = np.linspace(-500e-9, 0, num_pts)


def max_coordinates(coordinates, bead_diameter):
    abs_min = [abs(min(_c) - bead_diameter / 2) for _c in coordinates]
    abs_max = [abs(max(_c) + bead_diameter / 2) for _c in coordinates]
    return [max(_amin, _amax) for _amin, _amax in zip(abs_min, abs_max)]


# Obtain the integration order, we'll use the default "peirce" method, which performs a
# Gaussian-Legendre-like numerical integration over the back focal plane. Note that we need to
# include the bead radius, as the fields need to converge at the edge of the bead.
integration_order = objective.minimal_integration_order(
    max_coordinates([x, y, z], bead_diameter), method="peirce", lambda_vac=lambda_vac
)
print(f"integration order: {integration_order}")
# Obtain a function that returns the force on a bead at a certain coordinate when called.
force_func = trp.force_factory(
    partial(gaussian_beam, power=P, filling_factor=filling_factor),
    objective,
    bead,
    integration_order_bfp=integration_order,
    integration_method_bfp="peirce",
)

# %%
bead_center = [(0, 0, zz) for zz in z]
Fz = force_func(bead_center)[:, 2]

# %%
plt.figure(figsize=(8, 6))
plt.plot(z * 1e9, Fz * 1e12)
plt.xlabel("$z$ [nm]")
plt.ylabel("$F$ [pN]")
plt.title(f"{bead_diameter * 1e6} µm bead, $F_z$ at $x_b = y_b = 0$")
plt.show()

# %%
# linearly interpolate. For better accuracy, you might consider scipy.optimize.brentq
z_eval = interp1d(Fz, z)(0)
print(f"Force in z zero near z = {(z_eval*1e9):.1f} nm")

# %% [markdown]
# ## Forces in $x$ and $y$ directions
# Calculate the forces in the $x$ and $y$ direction at the location where the force in the $z$
# direction is (nearly) zero

# %%
x = np.linspace(-500e-9, 0, 201)
bead_center = [(xx, 0, z_eval) for xx in x]
Fx = force_func(bead_center)[:, 0]

# %%
y = np.linspace(-500e-9, 0, 201)
bead_center = [(0, yy, z_eval) for yy in y]
Fy = force_func(bead_center)[:, 1]

# %%
plt.figure(figsize=(8, 6))
plt.plot(x * 1e9, Fx * 1e12, label="x")
plt.plot(y * 1e9, Fy * 1e12, label="y")
plt.xlabel("displacement [nm]")
plt.ylabel("F [pN]")
plt.legend()
plt.title(
    f"{bead_diameter * 1e6} µm bead, $F_x$ at $(x, 0, {z_eval * 1e9:.1f})$ nm, $F_y$ at $(0, y, {z_eval * 1e9:.1f})$ nm"
)
plt.show()

# %% [markdown]
# ## Convergence
# ### To check accuracy:
# 1. Increase the order of the integration `integration_order` and check the difference between the old and newly calculated forces
# 1. Increase the number of samples at the back focal plane `bfp_sampling_n` and check the difference between the old and newly calculated forces
# 1. Increase the number of Mie scattering orders and check the difference between the old and newly calculated forces

# %%
lower_integration_order = integration_order - 7
Fz_less_precise = trp.forces_focus(
    partial(gaussian_beam, power=P, filling_factor=filling_factor),
    objective,
    bead=bead,
    integration_order_bfp=lower_integration_order,
    bead_center=[(0, 0, zz) for zz in z],
)[:, 2]


# %%
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(z * 1e9, Fz_less_precise * 1e12, label=f"Integration order = {lower_integration_order}")
plt.plot(z * 1e9, Fz * 1e12, label=f"Integration order = {integration_order}")
plt.xlabel("z [nm]")
plt.ylabel("F [pN]")
plt.legend()
plt.title(f"{bead_diameter * 1e6} µm bead, $F_z$ at X = Y = 0")
plt.subplot(1, 2, 2)
plt.plot(z * 1e9, (Fz - Fz_less_precise) * 1e12)
plt.xlabel("z [nm]")
plt.ylabel("difference [pN]")
plt.title("difference between calculations")
plt.show()

# %% [markdown]
# ### Gaining some speed
# 1. *Decrease* the number of spherical harmonics `num_spherical_modes` and check the difference between the old and newly calculated forces.
#     1. It may help to plot the absolute values of the Mie scattering coefficients on a logarithmic y axis to decide the initial cutoff.
# 1. *Decrease* the integration order, as done above, and check the difference between the previously and newly calculated forces.

# %%
an, bn = bead.ab_coeffs()
plt.figure(figsize=(8, 6))
plt.semilogy(range(1, an.size + 1), np.abs(an), label="$a_n$")
plt.semilogy(range(1, bn.size + 1), np.abs(bn), label="$b_n$")
plt.xlabel("Order")
plt.ylabel("|$a_n$|, $|b_n|$ [-]")
plt.title(f"Magnitude of scattering coefficients for {bead.bead_diameter * 1e6:.2f} µm bead")
plt.legend()
plt.grid()
plt.show()

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Calculating forces on a trapped bead

# %% tags=[]
# %matplotlib inline
import time, sys  # For progress bar
from IPython.display import clear_output  # For progress bar
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from pyoptics import mie_calc as mc
from scipy.interpolate import interp1d

font = {'weight' : 'normal',
        'size'   : 16}
rc('font', **font)

# %% [markdown] tags=[]
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

# %% tags=[]
bead_diameter = 4.4e-6  # [m]
lambda_vac = 1064e-9    # [m]
n_bead =  1.57          # [-]
n_medium = 1.33         # [-]

# %% tags=[]
# instantiate the MieCalc object
mie = mc.MieCalc(bead_diameter, n_bead, n_medium, lambda_vac)
# Tell use how many scattering orders are used according to the formula in literature:
print(f'Number of scattering orders used by default: {mie.number_of_orders()}')

# %% [markdown] tags=[]
# ## Properties of the objective
# See the image below. The definition of the coordinate system remains as before, and the objective is described by the $\mathit{NA}=n_\mathit{medium} \sin(\theta)$, the focal length $f$ in meters, and the medium at the back focal plane (BFP), $n_\mathit{bfp}$. The parameter `NA` sets the $\mathit{NA}$ (unitless), `focal_length` sets the focal length $f$ (in meters) and the refractive index of the medium at the BFP, $n_\mathit{bfp}$, is set by `n_bfp` (unitless).
# <figure>
#     <img src="images/objective.png" width=400>
#     <figcaption>Fig. 2: Coordinate system and the relation to the properties of the objective (NA, focal length, immersion medium)</figcaption>
# </figure>
#                                                                           

# %% tags=[]
# objective properties, for water immersion
NA = 1.2                # [-]
focal_length = 4.43e-3  # [m]
n_bfp = 1.0             # [-] Other side of the water immersion objective is air

# %% [markdown]
# ## Properties of the input beam
#
# The continuous field distribution at the back focal plane (the input beam, left figure) is discretized with $N$ samples, counting from the center to the extremes of the $x$ and $y$ axes (right figure, here $N=9$). The more samples you have, the more accurate the resulting point spread function will represent the continuous distribution. But, the calculation time will increase with $N^2$. You can get away with relatively small numbers as long as the distance between the bead and the focus is not too large. *Do* check for convergence, however (see below).
# <figure>
#     <img src="images/aperture_inf.png" width=400 align="left">
#     <img src="images/aperture_discretized.png" width=400>
#     <figcaption>Fig. 3: Left - continuous field distribution of the laser beam. Right - discretized approximation</figcaption>
# </figure>
#
# The parameter `bfp_sampling_n` determines the number of samples $N$ in the back focal plane.

# %%
# approximation of the focus, higher is better and slower (scales with N**2)
# best to check the correct range by plotting fields!
bfp_sampling_n=9

# 100% is 1.75W into a single trapping beam before the objective, at trap split = 50%
power_percentage = 25

# %%
filling_factor = 0.9                                # [-]
w0 = filling_factor * focal_length * NA / n_medium  # [m]
P = 1.75 * power_percentage / 100.                  # [W]
I0 = 2 * P / (np.pi * w0**2)                        # [W/m^2]
E0 = (I0 * 2/(mc._EPS0 * mc._C * n_bfp))**0.5       # [V/m]

def gaussian_beam(X_BFP, Y_BFP, R, Rmax, cosTheta, cosPhi, sinPhi): 
    Ex = np.exp(-(X_BFP**2 + Y_BFP**2) / w0**2) * E0
    return (Ex, None)


# %% [markdown]
# Add a little progress bar.
# Code from <https://www.mikulskibartosz.name/how-to-display-a-progress-bar-in-jupyter-notebook/>

# %%
def update_progress(progress):
    bar_length = 20
    progress = float(progress)
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


# %% [markdown]
# ## Forces in the $z$ direction - find the equilibrium
# Set the range in z to calculate the forces at. We expect the force on the bead in the z-direction to be zero for the interval of $[0, \inf)$
# Start with a range of $0\ldots 1 \mu m$

# %%
z = np.linspace(0, 1e-6, 20)  
Fz = np.empty(z.shape)

# %%
for idx, zz in enumerate(z):
    F = mie.forces_focused_fields(gaussian_beam, NA=NA, bfp_sampling_n=bfp_sampling_n, focal_length=focal_length, bead_center=(0, 0, zz), 
                                  num_orders=None, integration_orders=None, verbose=False)
    Fz[idx] = F[2]
    update_progress(idx / z.size)
update_progress(1.)

# %%
plt.figure(figsize=(8, 6))
plt.plot(z * 1e9, Fz * 1e12)
plt.xlabel('$z$ [nm]')
plt.ylabel('$F$ [pN]')
plt.title(f'{bead_diameter * 1e6} $\mu$m bead, $F_z$ at $x_b = y_b = 0$')
plt.show()

# %%
# linearly interpolate. For better accuracy, you might consider scipy.optimize.brentq
z_eval = interp1d(Fz, z)(0)
print(f'Force in z zero near z = {(z_eval*1e9):.1f} nm')

# %% [markdown]
# ## Forces in $x$ and $y$ directions
# Calculate the forces in the $x$ and $y$ direction at the location where the force in the $z$ direction is (nearly) zero

# %%
x = np.linspace(-500e-9, -1e-9, 21)
Fx = np.empty(x.shape)
for idx, xx in enumerate(x):
    F = mie.forces_focused_fields(gaussian_beam, NA=NA, bfp_sampling_n=bfp_sampling_n, focal_length=focal_length, 
                                  bead_center=(xx, 0, z_eval), num_orders=None, integration_orders=None)
    Fx[idx] = F[0]
    update_progress(idx / x.size)
update_progress(1.)

# %%
y = np.linspace(-500e-9, -1e-9, 21)
Fy = np.empty(y.shape)
for idx, yy in enumerate(y):
    F = mie.forces_focused_fields(gaussian_beam, NA=NA, bfp_sampling_n=bfp_sampling_n, focal_length=focal_length, 
                                  bead_center=(0, yy, z_eval), num_orders=None, integration_orders=None)
    Fy[idx] = F[1]
    update_progress(idx / y.size)
update_progress(1.)

# %%
plt.figure(figsize=(8, 6))
plt.plot(x * 1e9,Fx * 1e12, label='x')
plt.plot(y * 1e9,Fy * 1e12, label='y')
plt.xlabel('displacement [nm]')
plt.ylabel('F [pN]')
plt.legend()
plt.title(f'{bead_diameter * 1e6} $\mu$m bead, $F_x$ at $(x, 0, {z_eval * 1e9:.1f})$ nm, $F_y$ at $(0, y, {z_eval * 1e9:.1f})$ nm')
plt.show()

# %% [markdown]
# ## Convergence
# ### To check accuracy:
# 1. Increase the number of integration orders `integration_orders` and check the difference between the old and newly calculated forces
# 1. Increase the number of samples at the back focal plane `bfp_sampling_n` and check the difference between the old and newly calculated forces
# 1. Increase the number of Mie scattering orders and check the difference between the old and newly calculated forces

# %%
Fz1 = np.empty(z.size)
Fz2 = np.empty(z.size)
for idx, k in enumerate(z):
    F = mie.forces_focused_fields(gaussian_beam, NA=NA, bfp_sampling_n=bfp_sampling_n, 
                                  focal_length=focal_length, bead_center=(0, 0, k), 
                                  num_orders=None, integration_orders=None, verbose=False)
    Fz1[idx] = F[2]
    F = mie.forces_focused_fields(gaussian_beam, NA=NA, bfp_sampling_n=bfp_sampling_n * 2, 
                                  focal_length=focal_length, bead_center=(0, 0, k), 
                                  num_orders=None, integration_orders=None, verbose=False)
    Fz2[idx] = F[2]
    update_progress(idx / z.size)
update_progress(1.)

# %%
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(z * 1e9, Fz1 * 1e12, label=f'bfp_sampling = {bfp_sampling_n}')
plt.plot(z * 1e9, Fz2 * 1e12, label=f'bfp_sampling = {bfp_sampling_n * 2}')
plt.xlabel('z [nm]')
plt.ylabel('F [pN]')
plt.legend()
plt.title(f'{bead_diameter * 1e6} um bead, Fz at X = Y = 0')
plt.subplot(1, 2, 2)
plt.plot(z * 1e9, (Fz1 - Fz2) * 1e12)
plt.xlabel('z [nm]')
plt.ylabel('difference [pN]')
plt.title('difference between calculations')
plt.show()

# %% [markdown]
# ### Gaining some speed
# 1. *Decrease* the number of spherical harmonics `num_orders` and check the difference between the old and newly calculated forces.
#     1. It may help to plot the absolute values of the Mie scattering coefficients on a logarithmic y axis to decide the initial cutoff:
# 1. *Decrease* the number of plane waves in the back focal plane, and check the difference between the old and newly calculated forces.

# %%
an, bn = mie.ab_coeffs()
plt.figure(figsize=(8, 6))
plt.semilogy(range(1, an.size + 1), np.abs(an), label='$a_n$')
plt.semilogy(range(1, bn.size + 1), np.abs(bn), label='$b_n$')
plt.xlabel('Order')
plt.ylabel('|$a_n$|, $|b_n|$ [-]')
plt.legend()
plt.grid()
plt.show()

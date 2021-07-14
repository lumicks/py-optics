# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=[]
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from pyoptics import mie_calc as mc

# %% [markdown]
# # Examples of spherical particle in a focus
# **Note:** These examples are optimized to give a quick result, not an accurate result. Generally speaking, the parameter `bfp_sampling_n` should be increased considerably for an accurate representation of the focus. 

# %% [markdown]
# ## 1 micron polystyrene particle in water
# Laser at 1064 nm

# %%
bead_size = 1.0e-6
lambda_vac = 1.064e-6
n_bead =  1.5
n_medium = 1.33
num_pts = 41
x = np.linspace(-1*bead_size, 1*bead_size, num_pts)
z = np.linspace(-1*bead_size, 1*bead_size, num_pts)
mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)

# %%
Ex, Ey, Ez, X, Y, Z = mie.fields_gaussian_focus(n_bfp=1.0, focal_length=4.43e-3, NA=1.2, x=x, y=0, z=z, 
                                         bead_center=(0,0,0), bfp_sampling_n=9, return_grid=True)

# %%
plt.figure(figsize=(10,8))
plt.pcolor(Z,X, np.sqrt(np.abs(Ez)**2+np.abs(Ex)**2), cmap='plasma', shading='auto')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.title('|E|')
plt.colorbar()
plt.show()

# %% [markdown]
# Imaginary part of the z-component of the E field:

# %% tags=[]
plt.figure(figsize=(10,8))
plt.pcolor(Z,X, np.imag(Ez), cmap='jet', shading='auto')
plt.colorbar()
plt.title('imag(Ez)')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.show()

# %% [markdown]
# ## 4.05 mu bead 'kissing' the focus
# Shows bead off-center from focus, bead in air, NA of 0.9. Also, a cross section over the x-plane is taken. Because of symmetry, Ez and Ey should be zero in this plane. 

# %%
bead_size = 4.05e-6
lambda_vac = 1.064e-6
n_bead =  1.5
n_medium = 1.0
num_pts = 81
y = np.linspace(-0.5*bead_size, 1.5*bead_size, num_pts)
z = np.linspace(-1*bead_size, 1*bead_size, num_pts)
mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)

# %%
Ex, Ey, Ez, X, Y, Z = mie.fields_gaussian_focus(n_bfp=1.0, focal_length=4.43e-3, NA=.9, x=0, y=y, z=z, 
                                         bead_center=(0,2.9e-6,0), bfp_sampling_n=9, return_grid=True)

# %%
plt.figure(figsize=(10,8))
plt.pcolor(Z,Y, np.sqrt(np.abs(Ez)**2+np.abs(Ex)**2), cmap='plasma', shading='auto')
plt.colorbar()
plt.title('|E|')
plt.xlabel('Z [m]')
plt.ylabel('Y [m]')
plt.show()

# %% [markdown]
# Imaginary part of Ex. Modify the code to check that Ez and Ey are practically zero too:

# %% tags=[]
plt.figure(figsize=(10,8))
plt.pcolor(Z,Y, np.imag(Ex), cmap='jet', shading='auto')
plt.xlabel('Z [m]')
plt.ylabel('Y [m]')
plt.title('imag(Ex)')
plt.colorbar()
plt.show()

# %% [markdown]
# ## Silver nanoparticle, excited by 345 nm UV laser

# %%
bead_size = .2e-6
lambda_vac = 0.345e-6
n_bead =  0.06 + 1.76j # Silver at 345 nm
n_medium = 1.0
num_pts = 61
x = np.linspace(-1*bead_size, 1*bead_size, num_pts)
z = np.linspace(-1*bead_size, 1*bead_size, num_pts)
mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)

# %%
Ex, Ey, Ez, X, Y, Z = mie.fields_gaussian_focus(n_bfp=1.0, focal_length=4.43e-3, NA=.01, x=x, y=0, z=z, 
                                         bead_center=(0,0,0), bfp_sampling_n=9, return_grid=True)

# %%
plt.figure(figsize=(10,8))
plt.pcolor(Z,X, np.sqrt(np.abs(Ez)**2+np.abs(Ex)**2), cmap='plasma', shading='auto')
plt.colorbar()
plt.title('|E|')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.show()

# %% tags=[]
plt.figure(figsize=(10,8))
plt.pcolor(Z,X, np.imag(Ez), cmap='jet', shading='auto')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.title('imag(Ez)')
plt.colorbar()
plt.show()

# %% [markdown]
# ### Showcase a few other options

# %%
bead_size = 1e-6
lambda_vac = 1.064e-6
n_bead =  1.5
n_medium = 1.33
num_pts = 41
x = np.linspace(-1*bead_size, 1*bead_size, num_pts)
z = np.linspace(-1*bead_size, 1*bead_size, num_pts)
mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)

# %% [markdown]
# Limit the number of orders to 1 (dipole) by using the `num_orders` argument. Note the incorrect representation of the fields.
# Change `num_orders` to `None` or remove it to restore default behavior, which is to use the recommended number of orders from literature

# %%
Ex, Ey, Ez, X, Y, Z = mie.fields_gaussian_focus(n_bfp=1.0, focal_length=4.43e-3, NA=.01, x=x, y=0, z=z, 
                                         bead_center=(0,0,0), num_orders=1,
                                         bfp_sampling_n=9, return_grid=True)

# %%
plt.figure(figsize=(10,8))
plt.pcolor(Z,X, np.sqrt(np.abs(Ez)**2+np.abs(Ex)**2), cmap='plasma', shading='auto')
plt.colorbar()
plt.title('|E|')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.show()

# %% [markdown]
# Ask how many orders are necessary according to the literature, then use half

# %%
num_ord = mie.number_of_orders()
print(f'Number of orders is {num_ord}')
Ex, Ey, Ez, X, Y, Z = mie.fields_gaussian_focus(n_bfp=1.0, focal_length=4.43e-3, NA=.01, x=x, y=0, z=z, 
                                         bead_center=(0,0,0), num_orders=np.max((num_ord//2,1)),
                                         bfp_sampling_n=9, return_grid=True)

# %%
plt.figure(figsize=(10,8))
plt.pcolor(Z,X, np.sqrt(np.abs(Ez)**2+np.abs(Ex)**2), cmap='plasma', shading='auto')
plt.colorbar()
plt.title('|E|')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.show()

# %% [markdown]
# Get the scattering coefficients $a_n$ and $b_n$

# %%
mie.ab_coeffs()

# %% [markdown]
# Only calculate the external field, and only the scattered field (so not the focused laser). Change the bead from small to larger to see the transitions from Rayleigh scattering to forward scattering

# %%
bead_size = 0.1e-6
lambda_vac = 1.064e-6
n_bead =  1.5
n_medium = 1.33
num_pts = 41
x = np.linspace(-2*bead_size, 2*bead_size, num_pts)
z = np.linspace(-2*bead_size, 2*bead_size, num_pts)
mie = mc.MieCalc(bead_size, n_bead, n_medium, lambda_vac)

# %%
Ex, Ey, Ez, X, Y, Z = mie.fields_gaussian_focus(n_bfp=1.0, focal_length=4.43e-3, NA=.9, x=x, y=0, z=z, 
                                         bead_center=(0,0,0), num_orders=None, total_field=False, inside_bead=False,
                                         bfp_sampling_n=9, return_grid=True)

# %%
plt.figure(figsize=(10,8))
plt.pcolor(Z,X, np.sqrt(np.abs(Ez)**2+np.abs(Ex)**2), cmap='plasma', shading='auto')
plt.colorbar()
plt.title('|E| (scattered)')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.show()

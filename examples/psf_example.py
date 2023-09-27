# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import lumicks.pyoptics.psf as psf
from lumicks.pyoptics.psf.reference import focused_gauss_ref
from lumicks.pyoptics.psf.direct import focused_gauss

# %% [markdown]
# # Visual comparison
# Calculate a reference electric field and a CZT-based one in the focal plane
# #### Reference:

# %%
numpoints = 81
xy_range = (-5e-6, 5e-6)
z = 0
x_range = np.linspace(xy_range[0], xy_range[1], numpoints)

Ex_ref, Ey_ref, Ez_ref = focused_gauss_ref(lambda_vac=1064e-9, n_bfp=1.0, n_medium=1.33, 
                                               focal_length=4.43e-3, filling_factor=0.9, NA=1.2, x=x_range, y=x_range,z=z)

# %% [markdown]
# #### CZT-based calculation
# Change `bfp_sampling_n` below to see aliasing (`bfp_sampling_n=5`), or to largely suppress it in this area (`bfp_sampling_n=30`)

# %%
Ex, Ey, Ez, X, Y, Z = psf.fast_gauss(lambda_vac=1064e-9, n_bfp=1.0, n_medium=1.33, 
                                        focal_length=4.43e-3, filling_factor=0.9, NA=1.2, 
                                        x_range=xy_range, y_range=xy_range, z=z, 
                                         numpoints_x=numpoints, numpoints_y=numpoints,
                                         bfp_sampling_n=5, return_grid=True)

# %% [markdown]
# Now plot the fields:

# %% jupyter={"source_hidden": true}
for field, title in ((Ex,'Ex plane wave'),(Ex_ref,'Ex ground truth'),(Ey,'Ey plane wave'),(Ey_ref,'Ey ground truth'),
                    (Ez,'Ez plane wave'),(Ez_ref,'Ez ground truth')):
    plt.figure(figsize=((6.5,3)))
    plt.subplot(1,2,1)
    plt.suptitle(title)
    plt.xlabel('x [µm]')
    plt.ylabel('y [µm]')
    plt.pcolormesh(X * 1e6, Y * 1e6, np.real(field), shading='auto')
    plt.subplot(1,2,2)
    plt.pcolormesh(X * 1e6, Y * 1e6, np.imag(field), shading='auto')
    plt.xlabel('x [µm]')
    plt.show()
plt.close('all')


# %% [markdown]
# And plot the differences:

# %% jupyter={"source_hidden": true}
for field, title in [(Ex_ref-Ex, 'Ex'), (Ey_ref-Ey, 'Ey'), (Ez_ref-Ez, 'Ez')]:
    plt.figure(figsize=((8, 3)))
    plt.subplot(1,2,1)
    plt.suptitle(f'{title} diffence')
    plt.pcolormesh(X * 1e6, Y * 1e6, np.real(field), shading='auto')
    plt.colorbar()
    plt.xlabel('x [µm]')
    plt.ylabel('y [µm]')
    plt.subplot(1,2,2)
    plt.pcolormesh(X * 1e6, Y * 1e6, np.imag(field), shading='auto')
    plt.colorbar()
    plt.xlabel('x [µm]')
    plt.show()
plt.close('all')


# %% [markdown]
# ## Numerical comparison 
# Take cross sections along the optical axis (z), at random (x,y) points. Plot the field components and compare the plane wave spectrum method with the ground truth. Repeat at will

# %%
numpoints = 81
xy_range = 2e-6
x_range = np.linspace(-xy_range, xy_range, numpoints)
point = np.random.standard_normal((2,))*2000e-9
x = point[0]
y = point[1]

Ex_ref, Ey_ref, Ez_ref = focused_gauss_ref(lambda_vac=1064e-9, n_bfp=1.0, n_medium=1.33, 
                                           focal_length=4.43e-3, filling_factor=0.9, NA=1.2, x=x, y=y,z=x_range)

# %% [markdown]
# Change `bfp_sampling_n` from 5 to 50 to 125, and see how that drastically brings the result closer to the ground truth

# %%
Ex, Ey, Ez = psf.fast_gauss(1064e-9, 1.0, 1.33, 4.43e-3, 0.9, 1.2, x_range=x, numpoints_x=1, y_range=y, numpoints_y=1, z=x_range, bfp_sampling_n=5)


# %% [markdown]
# Plot the field components:

# %%
for field_ref, field_czt, title in [(Ex_ref, Ex, 'Ex'), (Ey_ref, Ey, 'Ey'), (Ez_ref, Ez, 'Ez')]:
    plt.figure(figsize=(15,3))
    plt.subplot(1,2,1)
    plt.plot(x_range, field_ref.real, label='ref')
    plt.plot(x_range, field_czt.real, label='plane wave')
    plt.xlabel('z [m]')
    plt.ylabel('E [V/m]')
    plt.gca().set_title(f'real({title})')
    plt.legend()
    plt.subplot(1,2,2)
    plt.gca().set_title(f'imag({title})')
    plt.plot(x_range, field_ref.imag, label='ref')
    plt.plot(x_range, field_czt.imag, label='plane wave')
    plt.xlabel('z [m]')
    plt.ylabel('E [V/m]')
    plt.legend()
    plt.show()
plt.close('all')

# %% [markdown]
# And plot the error:

# %%
for field1, field2, title in [(Ex_ref, Ex, 'Ex'), (Ey_ref, Ey, 'Ey'), (Ez_ref, Ez, 'Ez')]:
    plt.figure(figsize=(5,3))
    plt.plot(x_range, field1.real - field2.real, label='real')
    plt.plot(x_range, field1.imag - field2.imag, label='imag')
    plt.xlabel('z [m]')
    plt.ylabel('Error [V/m]')
    plt.legend()
    plt.gca().set_title(title)
    plt.show()
plt.close('all')

# %% [markdown]
# # Polarization
# Assert that x-polarized input gives the same result as y-polarized input, except for the obvious rotation of the fields. The graphs below should give a noisy output, close to 0, i.e., << 1e-8

# %%
numpoints = 81
dim = (-2.5e-6, 2.5e-6)
zrange = np.linspace(-5e-6,5e-6,81)
filling_factor = 1.0
NA = 1.2
focal_length = 4.43e-3
n_medium = 1.33
w0 = filling_factor * focal_length * NA / n_medium

def field_func_x(_, x_bfp, y_bfp, *args):
    # The first argument is not used
    Ein = np.exp(-((x_bfp)**2 + y_bfp**2)/w0**2)
    return (Ein, None)


def field_func_y(_, x_bfp, y_bfp, *args):
    # The first argument is not used
    Ein = np.exp(-((x_bfp)**2 + y_bfp**2)/w0**2)
    return (None, Ein)


Exc_x, Eyc_x, Ezc_x, Xx, Yx, Zx = psf.fast_psf(field_func_x, 1064e-9, 1.0, 1.33, 4.43e-3, 1.2, x_range=dim, numpoints_x=numpoints, 
                                           y_range=0, numpoints_y=1, z=zrange, bfp_sampling_n=125, return_grid=True)
Exc_y, Eyc_y, Ezc_y, Xy, Yy, Zy = psf.fast_psf(field_func_y, 1064e-9, 1.0, 1.33, 4.43e-3, 1.2, x_range=0, numpoints_x=1, 
                                           y_range=dim, numpoints_y=numpoints, z=zrange, bfp_sampling_n=125, return_grid=True)


# %%
for title, field1, field2 in [('X: $E_x$ - Y: $E_y$', Exc_x, Eyc_y), ('X: $E_y$ - Y: $E_x$', Eyc_x, Exc_y), ('X: $E_z$ - Y: $E_z$', Ezc_x, Ezc_y)]:
    plt.figure(figsize=((8, 3)))
    plt.subplot(1,2,1)
    plt.suptitle(title)
    plt.pcolor(Zx * 1e6, Xx * 1e6, np.real(field1 - field2), shading='auto')
    plt.xlabel('z [µm]')
    plt.ylabel('x [µm]')
    plt.gca().set_title('real')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.pcolor(Zx * 1e6, Xx * 1e6, np.imag(field1 - field2), shading='auto')
    plt.xlabel('z [µm]')
    plt.ylabel('x [µm]')
    plt.gca().set_title('imag')
    plt.colorbar()
    plt.show()
plt.close('all')

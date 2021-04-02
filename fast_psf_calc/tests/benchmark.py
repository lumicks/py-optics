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

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, sys, inspect

# %% jupyter={"source_hidden": true} tags=[]
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(
    inspect.getfile( inspect.currentframe() ))[0],"..")))
if cmd_subfolder not in sys.path:
     sys.path.insert(0, cmd_subfolder)
import psf_calc as psf

# %% jupyter={"source_hidden": true} tags=[]
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

# %% [markdown]
# # Visual comparison
# Calculate a reference electric field and a CZT-based one in the focal plane
# #### Reference:

# %%
numpoints = 81
xyrange = 5e-6
xrange = np.linspace(-xyrange, xyrange, numpoints)

Ex_ref, Ey_ref, Ez_ref = psf.focused_gauss_ref(lambda_vac=1064e-9, n_bfp=1.0, n_medium=1.33, 
                                               focal_length=4.43e-3, filling_factor=0.9, NA=1.2, x=xrange, y=xrange,z=0)

# %% [markdown]
# #### CZT-based calculation
# Change `bfp_sampling_n` below to see aliasing (`bfp_sampling_n=5`), or to largely suppress it in this area (`bfp_sampling_n=30`)

# %%
Ex, Ey, Ez, X, Y, Z = psf.fast_gauss_psf(lambda_vac=1064e-9, n_bfp=1.0, n_medium=1.33, 
                                        focal_length=4.43e-3, filling_factor=0.9, NA=1.2, 
                                        xrange=xyrange*2, yrange=xyrange*2, z=0, 
                                         numpoints_x=numpoints, numpoints_y=numpoints,
                                         bfp_sampling_n=5, return_grid=True)

# %% [markdown]
# Now plot the fields:

# %% tags=[] jupyter={"source_hidden": true}
for field, title in ((Ex,'Ex plane wave'),(Ex_ref,'Ex ground truth'),(Ey,'Ey plane wave'),(Ey_ref,'Ey ground truth'),
                    (Ez,'Ez plane wave'),(Ez_ref,'Ez ground truth')):
    plt.figure(figsize=((12,6)))
    plt.subplot(1,2,1)
    plt.suptitle(title)
    plt.pcolor(X, Y, np.real(field), shading='auto')
    plt.subplot(1,2,2)
    plt.pcolor(X, Y, np.imag(field), shading='auto')
    plt.show()


# %% [markdown]
# And plot the differences:

# %% jupyter={"source_hidden": true} tags=[]
for field, title in [(Ex_ref-Ex, 'Ex'), (Ey_ref-Ey, 'Ey'), (Ez_ref-Ez, 'Ez')]:
    plt.figure(figsize=((15,6)))
    plt.subplot(1,2,1)
    plt.suptitle(f'{title} diffence')
    plt.pcolor(X, Y, np.real(field), shading='auto')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.pcolor(X, Y, np.imag(field), shading='auto')
    plt.colorbar()
    plt.show()


# %% [markdown]
# ## Numerical comparison 
# Take cross sections along the optical axis (z), at random (x,y) points. Plot the field components and compare the plane wave spectrum method with the ground truth. Repeat at will

# %%
numpoints = 81
xyrange = 2e-6
xrange = np.linspace(-xyrange, xyrange, numpoints)
point = np.random.standard_normal((2,))*2000e-9
x = point[0]
y = point[1]

Ex_ref, Ey_ref, Ez_ref = psf.focused_gauss_ref(lambda_vac=1064e-9, n_bfp=1.0, n_medium=1.33, 
                                               focal_length=4.43e-3, filling_factor=0.9, NA=1.2, x=x, y=y,z=xrange)

# %% [markdown]
# Change `bfp_sampling_n` from 5 to 50 to 125, and see how that drastically brings the result closer to the ground truth

# %%
Ex, Ey, Ez = psf.focused_gauss(1064e-9, 1.0, 1.33, 4.43e-3, 0.9, 1.2, x, y, xrange, bfp_sampling_n=5)

# %% [markdown]
# Plot the field components:

# %% tags=[]
for field_ref, field_czt, title in [(Ex_ref, Ex, 'Ex'), (Ey_ref, Ey, 'Ey'), (Ez_ref, Ez, 'Ez')]:
    plt.figure(figsize=(20,4))
    plt.subplot(1,2,1)
    plt.plot(xrange, field_ref.real, label='ref')
    plt.plot(xrange, field_czt.real, label='czt')
    plt.xlabel('z [m]')
    plt.ylabel('E [V/m]')
    plt.gca().set_title(f'real({title})')
    plt.legend()
    plt.subplot(1,2,2)
    plt.gca().set_title(f'imag({title})')
    plt.plot(xrange, field_ref.imag, label='ref')
    plt.plot(xrange, field_czt.imag, label='czt')
    plt.xlabel('z [m]')
    plt.ylabel('E [V/m]')
    plt.legend()
    plt.show()

# %% [markdown]
# And plot the error:

# %% tags=[]
for field1, field2, title in [(Ex_ref, Ex, 'Ex'), (Ey_ref, Ey, 'Ey'), (Ez_ref, Ez, 'Ez')]:
    plt.figure(figsize=(8,6))
    plt.plot(xrange, field1.real - field2.real, label='real')
    plt.plot(xrange, field1.imag - field2.imag, label='imag')
    plt.xlabel('z [m]')
    plt.ylabel('Error [V/m]')
    plt.legend()
    plt.gca().set_title(title)
    plt.show()

# %% [markdown]
# # Polarization
# Assert that x-polarized input gives the same result as y-polarized input, except for the obvious rotation of the fields. The graphs below should give a noisy output, close to 0, i.e., << 1e-8

# %%
numpoints = 81
dim = 5e-6
zrange = np.linspace(-5e-6,5e-6,81)
filling_factor = 1.0
NA = 1.2
focal_length = 4.43e-3
n_medium = 1.33
w0 = filling_factor * focal_length * NA / n_medium

def field_func_x(X_BFP, Y_BFP, *args):
    Ein = np.exp(-((X_BFP)**2 + Y_BFP**2)/w0**2)
    return (Ein, None)


def field_func_y(X_BFP, Y_BFP, *args):
    Ein = np.exp(-((X_BFP)**2 + Y_BFP**2)/w0**2)
    return (None, Ein)


Exc_x, Eyc_x, Ezc_x, Xx, Yx, Zx = psf.fast_psf_calc(field_func_x, 1064e-9, 1.0, 1.33, 4.43e-3, 1.2, xrange=dim, numpoints_x=numpoints, 
                                           yrange=0, numpoints_y=1, z=zrange, bfp_sampling_n=125, return_grid=True)
Exc_y, Eyc_y, Ezc_y, Xy, Yy, Zy = psf.fast_psf_calc(field_func_y, 1064e-9, 1.0, 1.33, 4.43e-3, 1.2, xrange=0, numpoints_x=1, 
                                           yrange=dim, numpoints_y=numpoints, z=zrange, bfp_sampling_n=125, return_grid=True)


# %%
for title, field1, field2 in [('X: Ex - Y:Ey', Exc_x, Eyc_y), ('X: Ey - Y:Ex', Eyc_x, Exc_y), ('X: Ez - Y:Ez', Ezc_x, Ezc_y)]:
    plt.figure(figsize=((15,6)))
    plt.subplot(1,2,1)
    plt.suptitle(title)
    plt.pcolor(Zx, Xx, np.real(field1 - field2), shading='auto')
    plt.xlabel('z [m]')
    plt.ylabel('x [m]')
    plt.gca().set_title('real')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.pcolor(Zx, Xx, np.imag(field1 - field2), shading='auto')
    plt.xlabel('z [m]')
    plt.ylabel('x [m]')
    plt.gca().set_title('imag')
    plt.colorbar()
    plt.show()

# %%

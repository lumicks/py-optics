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

# %% [markdown]
# # Optical forces and the breakdown of the dipole approximation
#
# The dipole approximation treats a bead as an infinitely small particle with a dipole moment $\mathbf{p}$. When this approximation is valid, calculating forces on a bead is fast and convenient. In that case, the force on the particle is proportional to the dipole moment $\mathbf{p}$ and the gradient of the electric field [[1]](#1):
#
# $$F_i = 1 / 2\, \mathrm{Re}\, (p_i^* \nabla E_i),\, i \in x, y, z$$
#
# We assume that the dipole moment induced by the electric field is linear with the electric field itself and shows no anisotropy[<sup id="bh_def_rev">1</sup>](#bh_def):
#
# $$\mathbf{p} = \alpha \mathbf{E} $$
#
# where we have introduced the scalar polarizability of the particle $\alpha$. Then, the force is obtained from [[1]](#1)
#
# $$F_i = \alpha' / 2\, \mathrm{Re}\, \{E_i^* \nabla E_i\} + \alpha'' / 2\, \mathrm{Im}\, \{E_i^* \nabla E_i\}$$
#
# with $\alpha = \alpha' + i\alpha''$
#
# In short: in order to calculate the force on a dipole, only the gradient of the electric field it is in and the polarizability of the particle need to be known. However, the approximation breaks down in a range where optical traps are often used. We will check the applicability of the dipole approximation in the example below.
#
# [<sup id="#bh_def">1</sup>](#bh_def_rev) Note that Ref. [2] defines $\mathbf{p}$ as $\mathbf{p} = \varepsilon_m \alpha \mathbf{E}$, and therefore the definition of $\alpha$ there is slightly modified compared to below</span>

# %%
import numpy as np
from scipy.constants import epsilon_0
from scipy.constants import speed_of_light as C
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import lumicks.pyoptics.trapping as trp
import lumicks.pyoptics.psf as psf
from tqdm.auto import tqdm

# %% [markdown]
# ## Case 1: small bead, approximation valid
# Let's look at a polystyrene bead of 100 nm diameter, in an optical trap driven by a laser of $\lambda_0 = 1064$ nm. In that case, the bead is about 10x smaller than the wavelength, and as we'll see, the dipole approximation is quite accurate.

# %%
bead_diameter = 0.1e-6  # 100 nm bead
lambda_vac = 1064e-9  # 1064 nm laser
n_bead = 1.57  # polystyrene
n_medium = 1.33  # water

# %%
# instantiate a Bead object
small_bead = trp.Bead(
    bead_diameter=bead_diameter, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac
)
print(small_bead)


# %% [markdown]
# The polarizability of a small bead can be expressed as a function of the refractive index of the particle $n_b$, of the medium $n_m$, and of the size of the bead (radius $r$). A common *approximation* is to use the quasi-static approach, which assumes that the electric field inside the particle is homogeneous. This is a fair approximation as long as the particle is much smaller than the wavelength of the light. In that case [[2]](#2):
#
# $$\alpha_s = 4 \pi \epsilon_0 n_m^2 r^3 (n_b^2 - n_m^2) / (n_b^2 + 2 n_m^2)$$
#
# Here, $\epsilon_0$ is the permittivity of free space. As even particles made from a perfectly transparent material scatter light, power of the incoming beam is redirected, and this is considered a loss. This is taken into account for the final polarizablility by using the radiation correction [[1]](#1):
#
# $$\alpha = \alpha_s + (i k^3 \alpha_s^2) / (6 \pi \epsilon_0 n_m^2)$$
#
# Here, $k = 2 \pi n_m / \lambda_0$

# %%
# get the polarizability based on the refractive index of a bead, the medium and the size of a bead
def polarizability(bead: trp.Bead):
    # quasi-static polarizability
    a_s = (
        4 * np.pi * epsilon_0 * bead.n_medium**2 * (bead.bead_diameter/2)**3
        * (bead.n_bead**2 - bead.n_medium**2)/(bead.n_bead**2 + 2 * bead.n_medium**2)
    )
    k = 2 * np.pi * bead.n_medium / bead.lambda_vac
    # correct for radiation reaction
    alpha = a_s + 1j * k**3 / (6 * np.pi * epsilon_0 * n_medium**2) * a_s**2
    return alpha


# %% [markdown]
# ### Objective
# Below we will define some properties of the objective used to exert the force on the particle. Commonly-used objective are of the water immersion type, and have a relatively high Numerical Aperture (NA) of around 1.2. Since only the front of the objective is immersed into water, we also need to specify the refractive index of the medium on the back side, which is typically air. For convenience we will take it to be the same as for vacuum, 1.0.

# %%
# objective properties, for water immersion
NA = 1.2  # [-]
focal_length = 4.43e-3  # [m]
n_bfp = 1.0  # [-] Other side of the water immersion objective is air
# Instantiate an Objective. Note that n_medium has to be defined here as well
objective = trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)
print(objective)

# %%
# Approximation of the focus, higher is better and slower (scales with N**2)
# Regardless, the approximate focus is a valid solution of the Maxwell Equations.
bfp_sampling_n = 9

# %%
Pmax = 1.75  # [W]
power_percentage = 5  # [%]

filling_factor = 0.9  # [-]
w0 = filling_factor * focal_length * NA / n_medium  # [m]
P = Pmax * power_percentage / 100.0  # [W]
I0 = 2 * P / (np.pi * w0**2)  # [W/m^2]
E0 = (I0 * 2 / (epsilon_0 * C * n_bfp)) ** 0.5  # [V/m]

# Field distribution of the incoming laser beam. Polarization is purely in the x-direction.
def gaussian_beam(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n):
    Ex = np.exp(-(x_bfp**2 + y_bfp**2) / w0**2).astype('complex128') * E0
    return (Ex, None)


# %%
# Define the cube where we want to calculate the forces. All distances are in [m]
numpoints = 81
dim_xy = 4e-6  # [m]
dim_z = 4e-6  # [m]
z = np.linspace(-dim_z / 2, dim_z / 2, numpoints) 
x = np.linspace(-dim_xy / 2, dim_xy / 2, numpoints)
y = np.linspace(-dim_xy / 2, dim_xy / 2, numpoints)

# %% [markdown]
# ### Visualization of the Point Spread function (PSF)
# Let's visualize the focus of the laser beam that will exert a force on our particle

# %%
Ex, Ey, Ez, X, Y, Z = psf.fast_psf(
        gaussian_beam, lambda_vac, objective.n_bfp, objective.n_medium, objective.focal_length, objective.NA,
        xrange=dim_xy, numpoints_x=numpoints, yrange=dim_xy, numpoints_y=numpoints, z=z,
        bfp_sampling_n=bfp_sampling_n, return_grid=True
)

# Calculate the intensity, which is proportional to |E|^2
I = (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)

slice = (numpoints - 1) // 2
fig, ax = plt.subplots(1, 2)
ax[0].set_aspect('equal', adjustable='box')
ax[0].pcolormesh(Z[:, slice, :] * 1e6, X[:, slice, :] * 1e6, I[:, slice, :])
ax[0].set_xlabel('z [μm]')
ax[0].set_ylabel('x [μm]')
ax[0].set_title('Slice at y = 0')
ax[1].set_aspect('equal', adjustable='box')
ax[1].pcolormesh(Z[slice, :, :] * 1e6, Y[slice, :, :] * 1e6, I[slice, :, :])
ax[1].set_xlabel('z [μm]')
ax[1].set_ylabel('y [μm]')
ax[1].set_title('Slice at x = 0')
plt.tight_layout()
plt.suptitle('Cross sections through point spread function')
plt.show()


# %% [markdown]
# Notice how the point spread function is slightly smaller in the $y$-direction than in the $x$-direction, a consequence of the polarized nature of light. Anticipating our results below, we would expect that the forces in the $y$-direction are slightly larger, as the gradient in that direction is steeper.
#
# ### Field gradients
# In order to get the field gradient $\nabla E_i,\, i \in (x, y, z)$, we need to take the derivative of each field component with respect to every coordinate. For example, $\nabla E_x = (\partial E_x/\partial x,\, \partial E_x/\partial y,\, \partial E_x/\partial z)$. Here we can use the fact that we can perform the differentation on the Gaussian beam *before* it enters the objective. As such, we only need to modify the field function, and in the focus we will get the desired derivatives.

# %%
def get_kx_ky_kz(focal_length, k, aperture, x_bfp, y_bfp, r_bfp, r_max):
    """Calculate the values of kx, ky and kz, properties of plane waves that make up the focus"""
    bfp_sampling_n = (x_bfp.shape[0] + 1) // 2
    sin_theta = r_bfp / focal_length
    cos_theta = np.ones_like(sin_theta)
    cos_theta[aperture] = (1 - sin_theta[aperture]**2)**0.5
    region = sin_theta > 0
    cos_phi = np.empty_like(sin_theta)
    sin_phi = np.empty_like(sin_theta)
    cos_phi[region] = x_bfp[region] / (focal_length * sin_theta[region])
    cos_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 1
    sin_phi[region] = y_bfp[region] / (focal_length * sin_theta[region])
    sin_phi[bfp_sampling_n - 1, bfp_sampling_n - 1] = 0
    sin_phi[np.logical_not(aperture)] = 0
    cos_phi[np.logical_not(aperture)] = 1
    kz = k * cos_theta
    kp = k * sin_theta
    kx = -kp * cos_phi
    ky = - kp * sin_phi
    return kx, ky, kz


def dEd_(field_func, focal_length, k, coordinate):
    """Wrap a `field_func` such that we get the derivative of the fields to `coordinate` in the focus"""
    def field_derivative(*args):
        # Takes the derivative of the fields to x, y or z in the focus
        kx, ky, kz = get_kx_ky_kz(focal_length, k, *args[:-1])
        Ex, Ey = field_func(*args)
        _k = {'x': kx, 'y': ky, 'z': kz}
            
        for E in (Ex, Ey):
            if E is not None:
                E *= 1j * _k[coordinate]
        return (Ex, Ey)
    return field_derivative


def dipole_force(alpha, z_pos, dim, numpoints):
    """Calculate the force on a dipole with polarizability `alpha`"""
    # Field functions to get the gradient of the electric field
    dEdx = dEd_(gaussian_beam, focal_length, 2 * np.pi * n_medium / lambda_vac, 'x')
    dEdy = dEd_(gaussian_beam, focal_length, 2 * np.pi * n_medium / lambda_vac, 'y')
    dEdz = dEd_(gaussian_beam, focal_length, 2 * np.pi * n_medium / lambda_vac, 'z')

    # Actual PSF
    Ex, Ey, Ez, X, Y, Z = psf.fast_psf(
        gaussian_beam, 1064e-9, 1.0, n_medium, 4.43e-3, 1.2, xrange=dim,
        numpoints_x=numpoints, yrange=dim, numpoints_y=numpoints, z=z_pos,
        bfp_sampling_n=bfp_sampling_n, return_grid=True
    )
    # Gradients at sampling points. Edx is [Exdx, Eydx, Ezdx]. Edy is [Exdy, Eydy, Ezdy] etc..
    Edx, Edy, Edz = [
        psf.fast_psf(
            dEd_, 1064e-9, 1.0, n_medium, 4.43e-3, 1.2, xrange=dim,
            numpoints_x=numpoints, yrange=dim, numpoints_y=numpoints, z=z_pos,
            bfp_sampling_n=bfp_sampling_n, return_grid=False)
        for dEd_ in (dEdx, dEdy, dEdz)
    ]

    # Ei* \nabla Ei
    E_grad_E_x = np.conj(Ex) * Edx[0] + np.conj(Ey) * Edx[1] + np.conj(Ez) * Edx[2]
    E_grad_E_y = np.conj(Ex) * Edy[0] + np.conj(Ey) * Edy[1] + np.conj(Ez) * Edy[2]
    E_grad_E_z = np.conj(Ex) * Edz[0] + np.conj(Ey) * Edz[1] + np.conj(Ez) * Edz[2]
    Fx, Fy, Fz = [
        np.real(alpha) / 2 * E_grad_E__.real + np.imag(alpha) / 2 * E_grad_E__.imag 
                  for E_grad_E__ in (E_grad_E_x, E_grad_E_y, E_grad_E_z)
    ]
    return Fx, Fy, Fz


# %% [markdown]
# ## Force on the dipole
# First we calculate the force on the bead in the dipole approximation. Since it is fast, we calculate the force on a grid of 81 x 81 x 81 grid points in one go.

# %%
alpha = polarizability(small_bead)
Fx_dipole, Fy_dipole, Fz_dipole = dipole_force(alpha, z, 4e-6, numpoints)

# %% [markdown]
# ## Forces in the $z$ direction
# We now calculate the force on the bead with the full electromagnetic treatment. Since this calculation is orders of magnitude slower, we will only calculate 41 points along the z-axis to find the equilibrium position

# %%
Fz_mie = np.empty(z.shape)
for idx, zz in enumerate(tqdm(z)):
    F = trp.forces_focus(
        gaussian_beam,
        objective,
        small_bead,
        bfp_sampling_n=bfp_sampling_n,
        bead_center=(0, 0, zz),
        num_orders=None,
        integration_orders=None,
        verbose=False,
    )
    Fz_mie[idx] = F[2]


# %% [markdown]
# ### Results
# Now let's plot the results of both calculations. We slice the cube of force data obtained with the dipole approximation to get the force in the z-direction on the optical axis (`z == 0`)

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(z * 1e6, Fz_mie * 1e12, label='Mie')
ax[0].plot(z * 1e6, Fz_dipole[(numpoints -1) // 2, (numpoints -1) // 2, :] * 1e12, label='dipole approximation')
ax[0].legend()
ax[0].set_ylabel('Force [pN]')
ax[1].plot(z * 1e6, (Fz_mie - Fz_dipole[(numpoints -1) // 2, (numpoints -1) // 2,:]) * 1e12, label='error')
ax[1].set_ylabel('$F_\mathit{Mie} - F_\mathit{dipole}$ [pN]')
ax[1].legend()
ax[1].set_xlabel('z [μm]')
plt.show()

# %% [markdown]
# The results are virtually identical, and the dipole approximation is working well. We continue by comparing the forces in the $x$- and $y$-directions:
#
# ## Forces in $x$ and $y$ directions
# We calculate the forces in the $x$ and $y$ direction at the location where the force in the $z$ direction is (nearly) zero, that is, near the equilibrium. We only need to calculate the forces with the full electromagnetic approach, as we already have the full solution everywhere in our volume for the dipole approximation.

# %%
Fx_mie = np.empty(x.shape)
for idx, xx in enumerate(tqdm(x, desc='Calculating Fx')):
    F = trp.forces_focus(
        gaussian_beam,
        objective,
        small_bead,
        bfp_sampling_n=bfp_sampling_n,
        bead_center=(xx, 0, 0),
        num_orders=None,
        integration_orders=None,
    )
    Fx_mie[idx] = F[0]

Fy_mie = np.empty(y.shape)
for idx, yy in enumerate(tqdm(y, desc='Calculating Fy')):
    F = trp.forces_focus(
        gaussian_beam,
        objective,
        small_bead,
        bfp_sampling_n=bfp_sampling_n,
        bead_center=(0, yy, 0),
        num_orders=None,
        integration_orders=None,
    )
    Fy_mie[idx] = F[1]


# %% [markdown]
# ### Plots
# We plot the forces in $x$ and $y$, obtained with both methods, and the error between the two:

# %%
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6,6))
ax[0].plot(x * 1e6, Fx_mie * 1e12, label='Mie - x')
ax[0].plot(x * 1e6, Fx_dipole[:, (numpoints - 1)  // 2, (numpoints - 1)  // 2] * 1e12, label='dipole approximation - x')
ax[0].plot(y * 1e6, Fy_mie * 1e12, label='Mie - y')
ax[0].plot(y * 1e6, Fy_dipole[(numpoints - 1)  // 2, :, (numpoints - 1)  // 2] * 1e12, label='dipole approximation - y')
ax[0].legend(loc='upper right')
ax[0].set_ylabel('Force [pN]')
ax[1].plot(z * 1e6, (Fx_mie - Fx_dipole[:, (numpoints - 1)  // 2, (numpoints - 1)  // 2]) * 1e12, label='error - x')
ax[1].plot(z * 1e6, (Fy_mie - Fy_dipole[(numpoints - 1)  // 2, :, (numpoints - 1)  // 2]) * 1e12, label='error - y')
ax[1].set_ylabel('$F_\mathit{Mie} - F_\mathit{dipole}$ [pN]')
ax[1].legend()
ax[1].set_xlabel('distance in x / y [μm]')
plt.show()


# %% [markdown]
# Again, the dipole approximation is working well for this case. Also, indeed the force in the $y$-direction is larger than in the $x$-direction, as expected based on the gradient of the point spread function. We now turn our attention to a bead size that is commonly used with optical trapping.
#
# ## Case 2: (too) large bead
#
# Here, we take a 1 µm bead, ten times larger than before, with the same refractive index as earlier, and in the same medium (water):

# %%
# instantiate a Bead object
bead_diameter = 1.0e-6  # [m]
medium_bead = trp.Bead(
    bead_diameter=bead_diameter, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac
)
print(medium_bead)

# %% [markdown]
# ## Force on the dipole
# Again, we first calculate the force on the bead in the dipole approximation. Since it is fast, we calculate the force on a grid of 81 x 81 x 81 grid points in one go.

# %%
alpha = polarizability(medium_bead)
Fx_dipole, Fy_dipole, Fz_dipole = dipole_force(alpha, z, 4e-6, numpoints)

# %% [markdown]
# ## Forces in the $z$ direction
# We now calculate the force on the bead with the full electromagnetic treatment. Since this calculation is orders of magnitude slower, we will only calculate 81 points along the $z$-axis to find the equilibrium position

# %%
Fz_mie = np.empty(z.shape)
for idx, zz in enumerate(tqdm(z)):
    F = trp.forces_focus(
        gaussian_beam,
        objective,
        medium_bead,
        bfp_sampling_n=bfp_sampling_n,
        bead_center=(0, 0, zz),
        num_orders=None,
        integration_orders=None,
        verbose=False,
    )
    Fz_mie[idx] = F[2]


# %% [markdown]
# ### Results
# Now let's plot the results of both calculations. We slice the cube of force data obtained with the dipole approximation to get the force in the $z$-direction on the optical axis (`z == 0`). Note that below, we plot the forces in separate graphs with a different scale on the $y$-axis!

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(z * 1e6, Fz_mie * 1e12, label='Mie')
ax[0].set_ylabel('Force [pN]')
ax[0].legend()
ax[1].plot(z * 1e6, Fz_dipole[(numpoints - 1) // 2, (numpoints - 1) // 2, :] * 1e12, label='dipole approximation', color='r')
ax[1].set_ylabel('Force [pN]')
ax[1].legend()
plt.show()

# %% [markdown]
# The forces obtained with the (inaccurate) dipole approximation are off by two orders of magnitude. Moreover, there is no stable trapping possible according to this result. The full electromagnetic treatment shows that forces are much lower than the dipole approximation predict, but also, that there is a stable trapping region just behind the focus, where the force in the $z$-direction changes sign.
#
# ## Forces in $x$ and $y$ directions
# For completeness, we also check the forces in the $x$- and $y$-direction.

# %%
Fx_mie = np.empty(x.shape)
for idx, xx in enumerate(tqdm(x, desc='Calculating Fx')):
    F = trp.forces_focus(
        gaussian_beam,
        objective,
        medium_bead,
        bfp_sampling_n=bfp_sampling_n,
        bead_center=(xx, 0, 0),
        num_orders=None,
        integration_orders=None,
    )
    Fx_mie[idx] = F[0]

Fy_mie = np.empty(y.shape)
for idx, yy in enumerate(tqdm(y, desc='Calculating Fy')):
    F = trp.forces_focus(
        gaussian_beam,
        objective,
        medium_bead,
        bfp_sampling_n=bfp_sampling_n,
        bead_center=(0, yy, 0),
        num_orders=None,
        integration_orders=None,
    )
    Fy_mie[idx] = F[1]


# %% [markdown]
# ### Plots
# We plot the forces in $x$ and $y$, obtained with both methods, and the error between the two:

# %%
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6,6))
ax[0].plot(x * 1e6, Fx_mie * 1e12, label='Mie - x')
ax[0].plot(x * 1e6, Fx_dipole[:, (numpoints - 1)  // 2, (numpoints - 1)  // 2] * 1e12, label='dipole approximation - x')
ax[0].plot(y * 1e6, Fy_mie * 1e12, label='Mie - y')
ax[0].plot(y * 1e6, Fy_dipole[(numpoints - 1)  // 2, :, (numpoints - 1)  // 2] * 1e12, label='dipole approximation - y')
ax[0].legend(loc='upper right')
ax[0].set_ylabel('Force [pN]')
ax[1].plot(z * 1e6, (Fx_mie - Fx_dipole[:, (numpoints - 1)  // 2, (numpoints - 1)  // 2]) * 1e12, label='error - x')
ax[1].plot(z * 1e6, (Fy_mie - Fy_dipole[(numpoints - 1)  // 2, :, (numpoints - 1)  // 2]) * 1e12, label='error - y')
ax[1].set_ylabel('$F_\mathit{Mie} - F_\mathit{dipole}$ [pN]')
ax[1].legend()
ax[1].set_xlabel('distance in x / y [μm]')
plt.show()

# %% [markdown]
# Also here, the forces are overestimated by the dipole approximation, and the approximation is not very useful.
#
# # References
# <a id="1">[1]</a> Novotny, L., & Hecht, B. (2012). *Principles of Nano-Optics (2nd ed.)*. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193
#
# <a id="2">[2]</a> Craig F. Bohren & Donald R. Huffman (1983). *Absorption and Scattering of Light by Small Particles*. WILEY‐VCH Verlag GmbH & Co. KGaA. doi:10.1002/9783527618156

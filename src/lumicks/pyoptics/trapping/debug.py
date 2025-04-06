import numpy as np

from pyinstrument import Profiler
from scipy.constants import epsilon_0
from scipy.constants import speed_of_light as C

import lumicks.pyoptics.trapping as trp

# from tqdm import tqdm


# set_num_threads(1)
bead_diameter = 4.4e-6  # [m]
lambda_vac = 1064e-9  # [m]
n_bead = 1.57  # [-]
n_medium = 1.33  # [-]

# instantiate a Bead object
bead = trp.Bead(
    bead_diameter=bead_diameter, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac
)
# Tell use how many scattering orders are used according to the formula in literature:
print(f"Number of scattering orders used by default: {bead.number_of_orders}")

# objective properties, for water immersion
NA = 1.2  # [-]
focal_length = 4.43e-3  # [m]
n_bfp = 1.0  # [-] Other side of the water immersion objective is air
# Instantiate an Objective. Note that n_medium has to be defined here as well
objective = trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)
condenser = trp.Objective(NA=1.4, focal_length=focal_length, n_bfp=1.0, n_medium=1.5)

# 100% is 1.75W into a single trapping beam before the objective, at trap split = 50%
Pmax = 1.75  # [W]
power_percentage = 0.001  # [%]

filling_factor = 0.9  # [-]
w0 = filling_factor * focal_length * NA / n_medium  # [m]
P = Pmax * power_percentage / 100.0  # [W]
I0 = 2 * P / (np.pi * w0**2)  # [W/m^2]
E0 = (I0 * 2 / (epsilon_0 * C * n_bfp)) ** 0.5  # [V/m]


def gaussian_beam(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n):
    Ex = np.exp(-(x_bfp**2 + y_bfp**2) / w0**2) * E0
    return (None, Ex)


def plane_wave(aperture, x_bfp, y_bfp, r_bfp, r_max, bfp_sampling_n):
    Ex = np.zeros(r_bfp.shape, dtype="complex128")
    Ex[r_bfp == 0] = 1
    return (None, Ex)


z = np.linspace(-2e-6, 1e-6, 2001)
Fz = np.empty(z.shape)

# prof = Profiler()
# prof.start()
# for idx, zz in enumerate(tqdm(z)):
#     F = trp.forces_focus(gaussian_beam, objective, bead, bfp_sampling_n=bfp_sampling_n, bead_center=(0, 0, zz),
#                                   num_orders=None, integration_orders=None, verbose=False)
#     Fz[idx] = F[2]
# prof.stop()
# prof.open_in_browser()

ff_func = trp.farfield_factory(
    gaussian_beam,
    objective,
    bead,
    objective_bfp_sampling_n=31,
    integration_order=None,
)
bfp_coords = objective.get_back_focal_plane_coordinates(bfp_sampling_n=31)
cos_theta, sin_theta, cos_phi, sin_phi, aperture = objective.get_farfield_cosines(bfp_coords)

# prof = Profiler()
# prof.start()
Et, Ep = [np.zeros_like(cos_theta, dtype="complex128") for _ in range(2)]
_Et, _Ep = ff_func(
    (0.0e-6, 0.0, 0.75e-6),
    cos_theta[aperture],
    sin_theta[aperture],
    cos_phi[aperture],
    sin_phi[aperture],
    condenser.focal_length,
    num_threads=4,
)
Et[aperture] = _Et
Ep[aperture] = _Ep
# prof.stop()
# prof.open_in_browser()

# bfp_fields = objective.sample_back_focal_plane(gaussian_beam, 15)

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(np.abs(Ep) ** 2 + np.abs(Et) ** 2)
plt.colorbar()
plt.show()
# prof = Profiler()
# prof.start()
# bead_center = [(0.0, 0.0, zz) for zz in z]
# for idx, zz in enumerate((z)):
#     F = force_func(bead_center=(0.0, 0.0, zz))
#     Fz[idx] = F[2]
# prof.stop()
# prof.open_in_browser()
# force_func((0,0,0))

# bead_center = [(0.0, 0.0, zz) for zz in z]
# prof = Profiler()
# prof.start()
# F = force_func(bead_center, num_threads=2)
# prof.stop()
# prof.open_in_browser()
# # Fz[idx] = F[2]
# plt.figure()
# plt.plot(z, F[:, 2])
# plt.show()

# def temp():
#     bfp = condenser.get_back_focal_plane_coordinates(condenser_bfp_sampling_n)
#     cos_theta_c, sin_theta_c, cos_phi, sin_phi, aperture = condenser.get_farfield_cosines(bfp)
#     # Adjust theta to the medium of the objective with Snell's law. We assume that the condenser is
#     # well-corrected, and only preservation of power in a ray and Fresnel losses are taken into
#     # account.
#     sin_theta = sin_theta_c * condenser.n_medium / objective.n_medium

#     # use the condenser's aperture as a hard stop, but do not evaluate plane waves that have an
#     # evanescent source in the sample either (|sin| > 1).
#     aperture = np.logical_and(np.abs(sin_theta) <= 1, aperture)

#     cos_theta = np.ones_like(sin_theta)
#     cos_theta[aperture] = (
#         (1 + sin_theta[aperture]) * (1 - sin_theta[aperture])
#     ) ** 0.5  # Correct for ray refraction, based on power balance:
#     field_correction_factor = (
#         objective.n_medium * cos_theta / (condenser.n_medium * cos_theta_c)
#     ) ** 0.5

#     # Fresnel losses at the interface
#     kz1 = cos_theta * objective.n_medium * 2 * np.pi / bead.lambda_vac
#     kz2 = cos_theta_c * condenser.n_medium * 2 * np.pi / bead.lambda_vac
#     t_p = tp(objective.n_medium, kz1, condenser.n_medium, kz2)
#     t_s = ts(kz1, kz2)

#     E_theta *= t_p * field_correction_factor
#         E_phi *= t_s * field_correction_factor
#         Ex, Ey, Ez = spherical_to_cartesian_from_angles(
#             cos_theta_c, sin_theta_c, cos_phi, sin_phi, 0.0, E_theta_all, E_phi_all
#         )
#         bfp = condenser.farfield_to_back_focal_plane_cosines(
#             [Ex, Ey, Ez], cos_theta_c, cos_phi, sin_phi
#         )

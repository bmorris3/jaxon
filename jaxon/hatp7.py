import numpy as np
import astropy.units as u
from astropy.constants import G

__all__ = []

planet_name = "HAT-P-7"

# Mansfield 2018
lines = """1.120–1.158 0.0334±0.0037
1.158–1.196 0.0413±0.0038
1.196–1.234 0.0404±0.0037
1.234–1.271 0.0501±0.0037
1.271–1.309 0.0503±0.0038
1.309–1.347 0.0498±0.0037
1.347–1.385 0.0530±0.0037
1.385–1.423 0.0510±0.0037
1.423–1.461 0.0547±0.0039
1.461–1.499 0.0621±0.0041
1.499–1.536 0.0607±0.0042
1.536–1.574 0.0593±0.0044
1.574–1.612 0.0594±0.0046
1.612–1.650 0.0593±0.0045""".splitlines()
central_wl = ([np.array(list(map(float, line.split(' ')[0].split('–')))).mean()
               for line in lines]) * u.um
depths = np.array([float(line.split(' ')[1].split('±')[0][:-1])
                   for line in lines]) * 1e-2
depths_err = np.array([float(line.split(' ')[1].split('±')[1][1:])
                       for line in lines]) * 1e-2

spitzer_wl = [3.6, 4.5]
spitzer_depth = 1e-2 * np.array([0.161, 0.186])
spitzer_depth_err = 1e-2 * np.array([0.014, 0.008])

# plt.plot(
#     central_wl, depths, 'o'
# )
# plt.plot(
#     spitzer_wl, spitzer_depth, 'o'
# )

kepler_mean_wl = [0.641]  # um
kepler_depth = [19e-6]  # eyeballed
kepler_depth_err = [10e-6]

all_depths = np.concatenate([depths, spitzer_depth])
all_depths_errs = np.concatenate([depths_err, spitzer_depth_err])
all_wavelengths = np.concatenate([central_wl.value, spitzer_wl])

rprs = float(1.431*u.R_jup / (2.00 * u.R_sun))

g = (G * u.M_jup / u.R_jup**2).to(u.cm/u.s**2).value

t0 = 2454954.357462  # Bonomo 2017
period = 2.204740    # Stassun 2017
rp = 16.9 * u.R_earth  # Stassun 2017
rstar = 1.991 * u.R_sun  # Berger 2017
a = 4.13 * rstar     # Stassun 2017
duration = 4.0398 / 24  # Holczer 2016
b = 0.4960           # Esteves 2015
rho_star = 0.27 * u.g / u.cm ** 3  # Stassun 2017
T_s = 6449           # Berger 2018

a_rs = float(a / rstar)
a_rp = float(a / rp)
rp_rstar = float(rp / rstar)
eclipse_half_dur = duration / period / 2

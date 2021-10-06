import numpy as np
from lightkurve import search_lightcurve
from astropy.stats import sigma_clip, mad_std
import astropy.units as u
from kelp import Filter
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx

from .utils import floatX
from .hatp7 import (
    planet_name, t0, period, eclipse_half_dur, b, rstar, rho_star, rp_rstar
)

__all__ = [
    'eclipse_model'
]

lcf = search_lightcurve(
    planet_name, mission="Kepler", cadence="long"
    # quarter=10
).download_all()

slc = lcf.stitch()

phases = ((slc.time.jd - t0) % period) / period
in_eclipse = np.abs(phases - 0.5) < 1.5 * eclipse_half_dur
in_transit = (
    (phases < 1.5 * eclipse_half_dur) |
    (phases > 1 - 1.5 * eclipse_half_dur)
)
out_of_transit = np.logical_not(in_transit)

slc = slc.flatten(
    polyorder=3, break_tolerance=10, window_length=1001, mask=~out_of_transit
).remove_nans()

phases = ((slc.time.jd - t0) % period) / period
in_eclipse = np.abs(phases - 0.5) < 1.5 * eclipse_half_dur
in_transit = (
    (phases < 1.5 * eclipse_half_dur) |
    (phases > 1 - 1.5 * eclipse_half_dur)
)
out_of_transit = np.logical_not(in_transit)

sc = sigma_clip(
    np.ascontiguousarray(slc.flux[out_of_transit], dtype=floatX),
    maxiters=100, sigma=8, stdfunc=mad_std
)

phase = np.ascontiguousarray(
    phases[out_of_transit][~sc.mask], dtype=floatX
)
time = np.ascontiguousarray(
    slc.time.jd[out_of_transit][~sc.mask], dtype=floatX
)

bin_in_eclipse = np.abs(phase - 0.5) < eclipse_half_dur
unbinned_flux_mean = np.mean(sc[~sc.mask].data)

unbinned_flux_mean_ppm = 1e6 * (unbinned_flux_mean - 1)
flux_normed = np.ascontiguousarray(
    1e6 * (sc[~sc.mask].data / unbinned_flux_mean - 1.0), dtype=floatX
)
flux_normed_err = np.ascontiguousarray(
    1e6 * slc.flux_err[out_of_transit][~sc.mask].value, dtype=floatX
)

filt = Filter.from_name("Kepler")
filt.bin_down(4)   # This speeds up integration by orders of magnitude
filt_wavelength, filt_trans = filt.wavelength.to(u.m).value, filt.transmittance


def eclipse_model():
    with pm.Model() as model:
        # Define a Keplerian orbit using `exoplanet`:
        orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=0, b=b, rho_star=rho_star.to(u.g / u.cm ** 3),
            r_star=float(rstar / u.R_sun)
        )

        # Compute the eclipse model (no limb-darkening):
        eclipse_light_curves = xo.LimbDarkLightCurve([0, 0]).get_light_curve(
            orbit=orbit._flip(rp_rstar), r=orbit.r_star,
            t=phase * period,
            texp=(30 * u.min).to(u.d).value
        )

        # Normalize the eclipse model to unity out of eclipse and
        # zero in-eclipse
        eclipse = 1 + pm.math.sum(eclipse_light_curves, axis=-1)

        eclipse_numpy = pmx.eval_in_model(eclipse)
    return eclipse_numpy

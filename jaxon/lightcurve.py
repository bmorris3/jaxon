import numpy as np
from lightkurve import search_lightcurve
from astropy.stats import sigma_clip, mad_std
import astropy.units as u
from kelp import Filter
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx

from .utils import floatX
from .hatp7 import get_planet_params

__all__ = [
    'eclipse_model',
    'get_filter',
    'get_light_curve'
]

cadence = "long"
cadence_duration = 30 * u.min


def get_light_curve(cadence=cadence):
    """
    Parameters
    ----------
    cadence : str {'long', 'short'}
        Kepler cadence mode
    """
    (planet_name, a_rs, a_rp, T_s, rprs, t0, period, eclipse_half_dur, b,
        rstar, rho_star, rp_rstar) = get_planet_params()

    lcf = search_lightcurve(
        planet_name, mission="Kepler", cadence=cadence
        # quarter=10
    ).download_all()

    slc = lcf.stitch()

    phases = ((slc.time.jd - t0) % period) / period
    in_transit = (
        (phases < 1.5 * eclipse_half_dur) |
        (phases > 1 - 1.5 * eclipse_half_dur)
    )
    out_of_transit = np.logical_not(in_transit)

    slc = slc.flatten(
        polyorder=3, break_tolerance=10, window_length=1001, mask=~out_of_transit
    ).remove_nans()

    phases = ((slc.time.jd - t0) % period) / period
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
    return phase, time, flux_normed, flux_normed_err


def get_filter():
    """
    Get the Kepler bandpass filter transmittance curve

    Returns
    -------
    filt_wavelength : numpy.ndarray
        Wavelengths in the transmittance curve
    filt_trans : numpy.ndarray
        Transmittances
    """
    filt = Filter.from_name("Kepler")
    filt.bin_down(4)   # This speeds up integration by orders of magnitude
    filt_wavelength, filt_trans = filt.wavelength.to(u.m).value, filt.transmittance
    return filt_wavelength, filt_trans


def eclipse_model(cadence_duration=cadence_duration):
    """
    Compute the (static) eclipse model

    Parameters
    ----------
    cadence_duration : astropy.unit.Quantity
        Exposure duration

    Return
    ------
    eclipse_numpy : numpy.ndarray
        Occultation vector normalized to unity out-of-eclipse and zero
        in-eclipse.
    """
    (planet_name, a_rs, a_rp, T_s, rprs, t0, period, eclipse_half_dur, b,
        rstar, rho_star, rp_rstar) = get_planet_params()
    phase, time, flux_normed, flux_normed_err = get_light_curve()

    with pm.Model():
        # Define a Keplerian orbit using `exoplanet`:
        orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=0, b=b, rho_star=rho_star.to(u.g / u.cm ** 3),
            r_star=float(rstar / u.R_sun)
        )

        # Compute the eclipse model (no limb-darkening):
        eclipse_light_curves = xo.LimbDarkLightCurve([0, 0]).get_light_curve(
            orbit=orbit._flip(rp_rstar), r=orbit.r_star,
            t=phase * period,
            texp=cadence_duration.to(u.d).value
        )

        # Normalize the eclipse model to unity out of eclipse and
        # zero in-eclipse
        eclipse = 1 + pm.math.sum(eclipse_light_curves, axis=-1)

        eclipse_numpy = pmx.eval_in_model(eclipse)
    return eclipse_numpy

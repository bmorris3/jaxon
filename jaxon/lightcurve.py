import numpy as np
from lightkurve import search_lightcurve, LightCurveCollection
from lightkurve.correctors import CBVCorrector
from astropy.stats import sigma_clip, mad_std
import astropy.units as u
from kelp import Filter
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx

from .utils import floatX
# from .hatp7 import get_planet_params
from .planets import get_planet_params

__all__ = [
    'eclipse_model',
    'get_filter',
    'get_light_curve'
]

cadence = "long"
cadence_duration = 30 * u.min
cbv_type = ['SingleScale']
cbv_indices = [np.arange(1, 9)]

def get_light_curve(planet_name, quarter=None, cadence=cadence):
    """
    Parameters
    ----------
    cadence : str {'long', 'short'}
        Kepler cadence mode
    """
    (planet_name, a_rs, a_rp, T_s, rprs, t0, period, eclipse_half_dur, b,
        rstar, rho_star, rp_rstar, mstar, mass) = get_planet_params(
        planet_name
    )

    lcf = search_lightcurve(
        planet_name, mission="Kepler", cadence=cadence,
        quarter=quarter
    ).download_all(flux_column='sap_flux')

    corrected_lcs = []
    for each_quarter_lc in lcf:
        cbvCorrector = CBVCorrector(each_quarter_lc)
        # Perform the correction
        cbvCorrector.correct_gaussian_prior(cbv_type=cbv_type,
                                            cbv_indices=cbv_indices,
                                            alpha=1e-4)
        corrected_lcs.append(cbvCorrector.corrected_lc)

    collection = LightCurveCollection(corrected_lcs)
    slc = collection.stitch().remove_nans()

    phases = ((slc.time.jd - t0) % period) / period
    in_transit = (
        (phases < 1.5 * eclipse_half_dur) |
        (phases > 1 - 1.5 * eclipse_half_dur)
    )
    out_of_transit = np.logical_not(in_transit)

    sc = sigma_clip(
        np.ascontiguousarray(slc.flux[out_of_transit], dtype=floatX),
        maxiters=100, sigma=5, stdfunc=mad_std
    )

    phase = np.ascontiguousarray(
        phases[out_of_transit][~sc.mask], dtype=floatX
    )
    time = np.ascontiguousarray(
        slc.time.jd[out_of_transit][~sc.mask], dtype=floatX
    )

    unbinned_flux_mean = np.mean(sc[~sc.mask].data)
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


def eclipse_model(
        planet_name, quarter=None, cadence_duration=cadence_duration
):
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
        rstar, rho_star, rp_rstar, mstar, mass) = get_planet_params(
        planet_name
    )
    phase, time, flux_normed, flux_normed_err = get_light_curve(
        planet_name, quarter=quarter
    )

    with pm.Model():
        # Define a Keplerian orbit using `exoplanet`:
        orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=0, b=b, rho_star=rho_star.to(u.g / u.cm ** 3),
            r_star=rstar
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

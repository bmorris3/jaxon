import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist
import jax
from jax import numpy as jnp
from celerite2.jax import GaussianProcess, terms
import arviz

from .utils import floatX, rng_key
from .reflected import reflected_phase_curve
from .thermal import thermal_phase_curve
from .tp import get_Tarr, polynomial_order, element_number, Parr, dParr
from .spectrum import (
    exojax_spectrum, res_vis, nus, wav, stellar_spectrum, stellar_spectrum_vis
)
from .hatp7 import (
    get_observed_depths, get_planet_params
)
from .lightcurve import (
    get_light_curve, eclipse_model, get_filter
)

__all__ = [
    'model',
    'run_mcmc',
    'get_model_kwargs'
]

(all_depths, all_depths_errs, all_wavelengths,
    kepler_mean_wl) = get_observed_depths()


def estimate_ellipsoidal_amplitude(mass, rstar, mstar, period):
    ellipsoidal_amplitude_estimate = (
        mass / 0.077 * rstar ** 3 * mstar ** -2 * period ** -2
    )
    return ellipsoidal_amplitude_estimate


def estimate_doppler_amplitude(mass, mstar, period):
    doppler_amplitude_estimate = (
        mass / 0.37 * mstar**(-2/3) * period**(-1/3)
    )
    return doppler_amplitude_estimate



def model(
        n_temps, phase, time, y, yerr, eclipse_numpy,
        filt_wavelength, filt_trans, a_rs, a_rp, T_s, rprs,
        mstar, mass, period, rstar, nus=nus, wav=wav, Parr=Parr, dParr=dParr,
        stellar_spectrum=stellar_spectrum,
        res=res_vis,
        predict=False
):
    """
    The full joint model passed to numpyro.

    Parameters
    ----------
    n_temps : int
        Number of temperatures in the T-P profile
    phase : numpy.ndarray
        Phase of the planetary orbit
    time : numpy.ndarray
        Time in BJD
    y : numpy.ndarray
        Normalized flux in ppm
    yerr : numpy.ndarray
        Normalized flux error in ppm
    eclipse_numpy : numpy.ndarray
        Normalized eclipse vector
    filt_wavelength : numpy.ndarray
        Filter transmittance wavelength array
    filt_trans : numpy.ndarray
        Filter transmittance array
    a_rs : float
        Semimajor axis normalized by stellar radius
    a_rp : float
        Semimajor axis normalized by the planetary radius
    T_s : float
        Stellar effective temperature
    rprs : float
        Radius ratio
    mstar : float
        Stellar mass in solar masses
    mass : float
        Planet mass in Jupiter masses
    period : float
        Orbital period [d]
    rstar : float
        Stellar radius in solar radii
    nus : numpy.ndarray
        Frequencies sampled in the spectrum
    wav : numpy.ndarray
        Wavelengths sampled in the spectrum
    Parr : numpy.ndarray
        Pressure array at each temperature in the T-P profile
    dParr : numpy.ndarray
        Delta pressure array at each temperature in the T-P profile
    stellar_spectrum_vis : numpy.ndarray
        Spectrum of the star
    res : float
        Spectral resolution
    predict : bool
        Turn on or off the gaussian process ``predict`` features
    """
    temps = numpyro.sample(
        "temperatures", dist.Uniform(low=500, high=5000),
        sample_shape=(n_temps,)
    )
    n_grid_points = 150
    phases_grid = jnp.linspace(0 + 0.01, 1 - 0.01, n_grid_points, dtype=floatX)
    xi_grid = jnp.linspace(-np.pi + 0.01, np.pi - 0.01, n_grid_points,
                           dtype=floatX)
    xi = 2 * np.pi * (phase - 0.5)

    # Define reflected light phase curve model according to
    # Heng, Morris & Kitzmann (2021)
    omega = numpyro.sample('omega', dist.Uniform(low=0, high=1))
    g = numpyro.sample('g',
        dist.TwoSidedTruncatedDistribution(
            dist.Normal(loc=0, scale=0.01),
            low=-0.1, high=0.1)
    )
    reflected_ppm_grid, A_g = reflected_phase_curve(
        phases_grid, omega, g, a_rp
    )
    # reflected_ppm = interpolate(phases_grid, reflected_ppm_grid, phase)
    reflected_ppm = jnp.interp(phase, phases_grid, reflected_ppm_grid)

    numpyro.deterministic('A_g', A_g)

    ellipsoidal_amp_estimate = estimate_ellipsoidal_amplitude(
        mass, rstar, mstar, period
    )
    doppler_amp_estimate = estimate_doppler_amplitude(
        mass, mstar, period
    )

    # Define the ellipsoidal variation parameterization (simple sinusoid)
    ellipsoidal_amp = numpyro.sample(
        'ellip_amp',
        dist.TwoSidedTruncatedDistribution(
            dist.Normal(
                loc=ellipsoidal_amp_estimate,
                scale=ellipsoidal_amp_estimate/4
            ), low=0, high=100
        )
    )
    ellipsoidal_model_ppm = - ellipsoidal_amp * jnp.cos(
        4 * np.pi * (phase - 0.5)) + ellipsoidal_amp

    # Define the doppler variation parameterization (simple sinusoid)
    doppler_amp = numpyro.sample(
        'doppler_amp',
        dist.TwoSidedTruncatedDistribution(
            dist.Normal(
                loc=doppler_amp_estimate,
                scale=doppler_amp_estimate/4
            ), low=0, high=10
        )
    )
    doppler_model_ppm = doppler_amp * jnp.sin(2 * np.pi * phase)

    # Define the thermal emission model according to description in
    # Morris et al. (in prep)
    n_phi = 150
    n_theta = 10
    phi = jnp.linspace(-2 * np.pi, 2 * np.pi, n_phi, dtype=floatX)
    theta = jnp.linspace(0, np.pi, n_theta, dtype=floatX)
    theta2d, phi2d = jnp.meshgrid(theta, phi)

    # ln_C_11_kepler = -2.6
    C_11_kepler = 0.35 # numpyro.sample('C_11', dist.Uniform(low=0, high=0.55))
    # hml_eps = numpyro.sample('epsilon', dist.Uniform(low=0, high=8 / 5))
    hml_f = 0.73 #(2 / 3 - hml_eps * 5 / 12) ** 0.25
    delta_phi = 0 #numpyro.sample(
    #     'delta_phi',
    #     dist.TwoSidedTruncatedDistribution(
    #         dist.Normal(loc=0, scale=0.05),
    #         low=-np.pi/4, high=np.pi/4
    #     )
    # )
    A_B = 0.0

    # Compute the thermal phase curve with zero phase offset
    thermal_grid, temp_map = thermal_phase_curve(
        xi_grid, delta_phi, 4.5, 0.6, C_11_kepler, T_s, a_rs, 1 / a_rp, A_B,
        theta2d, phi2d, filt_wavelength, filt_trans, hml_f
    )

    # thermal = interpolate(xi_grid, 1e6 * thermal_grid, xi)
    thermal = jnp.interp(xi, xi_grid, 1e6 * thermal_grid)

    # epsilon = 8 * nightside**4 / (3 * dayside**4 + 5 * nightside**4)
    # f = (2 / 3 - hml_eps * 5 / 12) ** 0.25

    # numpyro.deterministic('f', f)
    # numpyro.deterministic('epsilon', epsilon)

    # Define the composite phase curve model
    flux_norm = (eclipse_numpy *
        (reflected_ppm + thermal) + doppler_model_ppm + ellipsoidal_model_ppm
    )

    flux_norm -= jnp.mean(flux_norm)

    sigma = numpyro.sample(
        "sigma", dist.TwoSidedTruncatedDistribution(
            dist.Normal(loc=y.std(), scale=y.std()/10),
            low=0, high=1000 * y.ptp()
        )
    )
    kernel = terms.Matern32Term(sigma=sigma, rho=22)
    jitter = numpyro.sample('jitter', dist.Uniform(low=0, high=100))
    gp = GaussianProcess(kernel, mean=flux_norm)
    gp.compute(time, yerr=jnp.sqrt(yerr ** 2 + jitter ** 2),
               check_sorted=False)

    if predict:
        gp.condition(y)
        pred = gp.predict(y)
        numpyro.deterministic("therm", thermal)
        numpyro.deterministic("ellip", ellipsoidal_model_ppm)
        numpyro.deterministic("doppl", doppler_model_ppm)
        numpyro.deterministic("refle", reflected_ppm)
        numpyro.deterministic("model", flux_norm)
        numpyro.deterministic("resid", y - pred)
        numpyro.deterministic("pred", pred)

    # log_vmr_prod = numpyro.sample('log_vmr_prod',
    #                               dist.Uniform(low=-10, high=-4))
    vmr_prod = 1e-6
    mmr_TiO = 1e-6 #numpyro.sample("mmr_TiO", dist.Uniform(low=-9, high=-2))

    Tarr = get_Tarr(temps, Parr)
    Fcgs, _, _ = exojax_spectrum(
        temps, vmr_prod, mmr_TiO,
        Parr, dParr, nus, wav
    )

    fpfs_spectrum = rprs ** 2 * Fcgs / stellar_spectrum

    interp_depths = jnp.interp(
        all_wavelengths, wav / 1000, fpfs_spectrum
    )
    numpyro.deterministic("FpFs", fpfs_spectrum)
    numpyro.deterministic("interp_depths", interp_depths)
    numpyro.deterministic("Tarr", Tarr)

    kepler_thermal_eclipse_depth_obs = numpyro.deterministic(
        "kep_depth", jnp.interp(0, xi_grid, thermal_grid)
    )
    kepler_thermal_eclipse_depth_model = jnp.average(
        fpfs_spectrum,
        weights=(jnp.abs(kepler_mean_wl[0] - wav / 1000) < 0.3).astype(int)
    )

    numpyro.sample('phase_curve', gp.numpyro_dist(), obs=y)
    numpyro.factor('spectrum',
        dist.Normal(
            loc=interp_depths,
            scale=all_depths_errs
        ).log_prob(all_depths)
    )

    kepler_thermal_eclipse_depth_err = numpyro.sample(
        "kep_depth_err", dist.Uniform(low=1e-6, high=100e-6)
    )
    numpyro.factor(
        "obs_spectrum_kepler", dist.Normal(
            loc=kepler_thermal_eclipse_depth_model,
            scale=kepler_thermal_eclipse_depth_err
        ).log_prob(kepler_thermal_eclipse_depth_obs)
    )


def get_model_kwargs(quarter=None):
    phase, time, flux_normed, flux_normed_err = get_light_curve(quarter=quarter)

    filt_wavelength, filt_trans = get_filter()

    (planet_name, a_rs, a_rp, T_s, rprs, t0, period, eclipse_half_dur, b,
        rstar, rho_star, rp_rstar, mstar, mass) = get_planet_params()

    model_kwargs = dict(
        phase=phase.astype(floatX),
        time=(time - time.mean()).astype(floatX),
        y=flux_normed.astype(floatX),
        yerr=flux_normed_err.astype(floatX),
        eclipse_numpy=jnp.array(eclipse_model(quarter=quarter)).astype(floatX),
        filt_wavelength=jnp.array(filt_wavelength.astype(floatX)),
        filt_trans=jnp.array(filt_trans.astype(floatX)),
        a_rs=a_rs, a_rp=a_rp, T_s=T_s, period=period,
        mass=mass, mstar=mstar, rstar=rstar,
        n_temps=polynomial_order * element_number + 1,
        res=res_vis, rprs=rprs
    )
    return model_kwargs


def run_mcmc(run_title='tmp', num_warmup=5, num_samples=10, quarter=10):
    """
    Run MCMC with the NUTS via numpyro.

    Parameters
    ----------
    run_title : str
        Name of the run
    num_warmup : int
        Number of iterations in the burn-in phase
    num_samples : int
        Number of iterations of the sampler
    quarter : int, list of ints
        Kepler quarters to fit
    """
    model_kwargs = get_model_kwargs(quarter=quarter)

    print(f'Start MCMC, n chains = {len(jax.devices())}')
    mcmc = MCMC(
        sampler=NUTS(
            model, dense_mass=True,
        ),
        num_warmup=num_warmup,
        num_samples=num_samples,
        chain_method='parallel',
        num_chains=len(jax.devices()),
    )
    mcmc.run(
        rng_key,
        **model_kwargs
    )

    mcmc.post_warmup_state = mcmc.last_state
    print('Save first output')

    arviz_mcmc = arviz.from_numpyro(mcmc)
    arviz_mcmc.to_netcdf('chains_' + run_title + '0.nc')

import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist
import jax
from jax import numpy as jnp
from celerite2.jax import GaussianProcess, terms
import arviz

from .utils import floatX, rng_key
from .reflected import (
    reflected_phase_curve, reflected_phase_curve_inhomogeneous
)
from .thermal import thermal_phase_curve

from .planets import (
    get_planet_params
)
from .lightcurve import (
    get_light_curve, eclipse_model, get_filter
)

__all__ = [
    'model',
    'run_mcmc',
    'get_model_kwargs'
]


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
        phase, time, y, yerr, eclipse_numpy,
        filt_wavelength, filt_trans, a_rs, a_rp, T_s,
        mstar, mass, period, rstar, predict=False,
        dilution_factor=1,
        pointwise=False,
        **kwargs
):
    """
    The full joint model passed to numpyro.

    Parameters
    ----------
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
    n_grid_points = 150
    phases_grid = jnp.linspace(0 + 0.01, 1 - 0.01, n_grid_points, dtype=floatX)
    xi_grid = jnp.linspace(-np.pi + 0.01, np.pi - 0.01, n_grid_points,
                           dtype=floatX)
    xi = 2 * np.pi * (phase - 0.5)

    # Define reflected light phase curve model according to
    # Heng, Morris & Kitzmann (2021)
    omega0 = numpyro.sample('omega0', dist.Uniform(low=0, high=1))
    omega1 = numpyro.sample('omega1', dist.Uniform(low=0, high=1))
    omega_0 = jnp.sqrt(omega0) * omega1
    omega_prime = jnp.sqrt(omega0) * (1 - omega1)

    x1 = numpyro.sample('x1', dist.Uniform(low=-np.pi/2, high=0))
    x2 = numpyro.sample('x2', dist.Uniform(low=0, high=np.pi/2))

    A_g = numpyro.sample('A_g', dist.Uniform(low=0, high=0.6))

    reflected_ppm_grid, g, q = reflected_phase_curve_inhomogeneous(
        phases_grid, omega_0, omega_prime, x1, x2, A_g, a_rp
    )
    # reflected_ppm = interpolate(phases_grid, reflected_ppm_grid, phase)
    reflected_ppm = jnp.interp(phase, phases_grid, reflected_ppm_grid)

    numpyro.deterministic('g', g)

    numpyro.factor('g_factor',
        dist.TwoSidedTruncatedDistribution(
            dist.Normal(loc=0, scale=0.01),
            low=-0.1, high=0.1).log_prob(g)
    )

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
            ), low=0, high=50
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

    C_11_kepler = numpyro.sample('C_11',
        dist.TwoSidedTruncatedDistribution(
            dist.Normal(loc=0.2, scale=0.05),
            low=0, high=0.35)
    )
    hml_f = 0.73
    delta_phi = 0
    A_B = 0.0

    # Compute the thermal phase curve with zero phase offset
    thermal_grid, temp_map = thermal_phase_curve(
        xi_grid, delta_phi, 4.5, 0.6, C_11_kepler, T_s, a_rs, 1 / a_rp, A_B,
        theta2d, phi2d, filt_wavelength, filt_trans, hml_f
    )

    thermal = jnp.interp(xi, xi_grid, 1e6 * thermal_grid)

    # Define the composite phase curve model
    flux_norm = (eclipse_numpy *
        (reflected_ppm + thermal) +
        doppler_model_ppm + ellipsoidal_model_ppm
    ) / dilution_factor

    flux_norm -= jnp.mean(flux_norm)

    sigma = numpyro.sample(
        "sigma", dist.TwoSidedTruncatedDistribution(
            dist.Normal(loc=y.std(), scale=y.std()/10),
            low=0, high=10 * y.std()
        )
    )

    kernel = terms.Matern32Term(sigma=sigma, rho=30)
    jitter = numpyro.sample('jitter', dist.Uniform(low=0, high=1000))
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

    numpyro.sample('phase_curve', gp.numpyro_dist(), obs=y)

    if pointwise:
        diag = yerr ** 2 + jitter ** 2
        K_s = kernel.to_dense(time.flatten(), np.zeros_like(time))
        covariance_matrix = K_s + jnp.eye(*K_s.shape) * diag
        inv_cov = jnp.linalg.inv(covariance_matrix)

        # https://mc-stan.org/loo/articles/loo2-non-factorizable.html
        g_i = inv_cov @ (y - flux_norm)
        c_ii = jnp.diag(inv_cov)

        lnlike = (
            -0.5 * jnp.log(2 * np.pi) + 0.5 * jnp.log(c_ii) -
            0.5 * (g_i**2 / c_ii)
        )

        numpyro.deterministic("pointwise", lnlike)

def get_model_kwargs(planet_name, quarter=None):
    phase, time, flux_normed, flux_normed_err = get_light_curve(
        planet_name, quarter=quarter
    )

    filt_wavelength, filt_trans = get_filter()

    (planet_name, a_rs, a_rp, T_s, rprs, t0, period, eclipse_half_dur, b,
        rstar, rho_star, rp_rstar, mstar, mass) = get_planet_params(
        planet_name
    )

    if planet_name.startswith('KOI-13'):
        dilution_factor = 1.913
    else:
        dilution_factor = 1

    model_kwargs = dict(
        phase=phase.astype(floatX),
        time=(time - time.mean()).astype(floatX),
        y=flux_normed.astype(floatX),
        yerr=flux_normed_err.astype(floatX),
        eclipse_numpy=jnp.array(eclipse_model(
            planet_name, quarter=quarter)
        ).astype(floatX),
        filt_wavelength=jnp.array(filt_wavelength.astype(floatX)),
        filt_trans=jnp.array(filt_trans.astype(floatX)),
        a_rs=a_rs, a_rp=a_rp, T_s=T_s, period=period,
        mass=mass, mstar=mstar, rstar=rstar,
        dilution_factor=dilution_factor
    )
    return model_kwargs


def run_mcmc(
        planet_name, run_title='tmp', num_warmup=5, num_samples=10, quarter=10
):
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
    model_kwargs = get_model_kwargs(planet_name, quarter=quarter)

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

    # mcmc.post_warmup_state = mcmc.last_state

    arviz_mcmc = arviz.from_numpyro(mcmc)
    arviz_mcmc.to_netcdf('chains_' + planet_name + '_' + run_title + '0.nc')

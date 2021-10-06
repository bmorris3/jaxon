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
    exojax_spectrum, res_vis, nus, wav, bb_star_transformed
    #  cnu_TiO, indexnu_TiO
)
from .hatp7 import (
    rprs, all_depths, all_depths_errs, all_wavelengths, kepler_mean_wl,
    a_rs, a_rp, T_s
)
from .lightcurve import (
    phase, time, flux_normed, flux_normed_err, eclipse_model, filt_wavelength,
    filt_trans
)

__all__ = [
    'model',
    'run_mcmc'
]

model_kwargs = dict(
    phase=phase.astype(floatX),
    time=(time - time.mean()).astype(floatX),
    y=flux_normed.astype(floatX),
    yerr=flux_normed_err.astype(floatX),
    eclipse_numpy=jnp.array(eclipse_model()).astype(floatX),
    filt_wavelength=jnp.array(filt_wavelength.astype(floatX)),
    filt_trans=jnp.array(filt_trans.astype(floatX)),
    a_rs=a_rs, a_rp=a_rp, T_s=T_s, n_temps=polynomial_order * element_number + 1,
    res=res_vis
)


def model(
        n_temps, phase, time, y, yerr, eclipse_numpy,
        filt_wavelength, filt_trans, a_rs, a_rp, T_s,
        nus=nus, wav=wav, Parr=Parr, dParr=dParr,
        bb_star_transformed=bb_star_transformed,
        res=res_vis,
        predict=False
):
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
                           low=-0.1, high=0.1
                       )
                       )
    reflected_ppm_grid, A_g = reflected_phase_curve(
        phases_grid, omega, g, a_rp
    )
    # reflected_ppm = interpolate(phases_grid, reflected_ppm_grid, phase)
    reflected_ppm = jnp.interp(phase, phases_grid, reflected_ppm_grid)

    numpyro.deterministic('A_g', A_g)
    #     numpyro.deterministic('q', q)
    # Define the ellipsoidal variation parameterization (simple sinusoid)
    ellipsoidal_amp = numpyro.sample('ellip_amp',
                                     dist.Uniform(low=0, high=100))
    ellipsoidal_model_ppm = - ellipsoidal_amp * jnp.cos(
        4 * np.pi * (phase - 0.5)) + ellipsoidal_amp

    # Define the doppler variation parameterization (simple sinusoid)
    doppler_amp = numpyro.sample('doppler_amp', dist.Uniform(low=0, high=20))
    doppler_model_ppm = doppler_amp * jnp.sin(
        2 * np.pi * phase)

    # Define the thermal emission model according to description in
    # Morris et al. (in prep)
    # floatX = 'float32'
    n_phi = 75
    n_theta = 7
    phi = jnp.linspace(-2 * np.pi, 2 * np.pi, n_phi, dtype=floatX)
    theta = jnp.linspace(0, np.pi, n_theta, dtype=floatX)
    theta2d, phi2d = jnp.meshgrid(theta, phi)

    # ln_C_11_kepler = -2.6
    C_11_kepler = numpyro.sample('C_11', dist.Uniform(low=0, high=0.5))
    hml_eps = numpyro.sample('epsilon', dist.Uniform(low=0, high=8 / 5))
    hml_f = (2 / 3 - hml_eps * 5 / 12) ** 0.25
    delta_phi = numpyro.sample('delta_phi', dist.Uniform(low=-np.pi, high=0))

    A_B = 0.0

    # Compute the thermal phase curve with zero phase offset
    thermal_grid, temp_map = thermal_phase_curve(
        xi_grid, delta_phi, 4.5, 0.575, C_11_kepler, T_s, a_rs, 1 / a_rp, A_B,
        theta2d, phi2d, filt_wavelength, filt_trans, hml_f
    )

    # thermal = interpolate(xi_grid, 1e6 * thermal_grid, xi)
    thermal = jnp.interp(xi, xi_grid, 1e6 * thermal_grid)

    # epsilon = 8 * nightside**4 / (3 * dayside**4 + 5 * nightside**4)
    f = (2 / 3 - hml_eps * 5 / 12) ** 0.25

    numpyro.deterministic('f', f)
    # numpyro.deterministic('epsilon', epsilon)

    # Define the composite phase curve model
    flux_norm = (eclipse_numpy *
                 (reflected_ppm + thermal) +
                 doppler_model_ppm + ellipsoidal_model_ppm
                 )

    flux_norm -= jnp.mean(flux_norm)

    sigma = numpyro.sample(
        "sigma", dist.TwoSidedTruncatedDistribution(
            dist.Normal(loc=y.ptp(), scale=y.std()), low=0, high=4 * y.ptp()
        )
    )
    kernel = terms.Matern32Term(sigma=sigma, rho=22)
    jitter = 0  # numpyro.sample('jitter', dist.Uniform(low=0, high=y.ptp()))
    gp = GaussianProcess(kernel, mean=flux_norm)
    gp.compute(time, yerr=jnp.sqrt(yerr ** 2 + jitter ** 2),
               check_sorted=False)

    if predict:
        gp.condition(y)
        pred = gp.predict(time)
        numpyro.deterministic("therm", thermal)
        numpyro.deterministic("ellip", ellipsoidal_model_ppm)
        numpyro.deterministic("doppl", doppler_model_ppm)
        numpyro.deterministic("refle", reflected_ppm)
        numpyro.deterministic("model", flux_norm)
        numpyro.deterministic("resid", y - pred)
        numpyro.deterministic("pred", pred)

    log_vmr_prod = numpyro.sample('log_vmr_prod',
                                  dist.Uniform(low=-10, high=-4))

    mmr_TiO = numpyro.sample("mmr_TiO", dist.Uniform(low=-9, high=-2))

    Tarr = get_Tarr(temps, Parr)
    Fcgs, _, _ = exojax_spectrum(
        temps, jnp.power(10, log_vmr_prod), jnp.power(10, mmr_TiO),
        Parr, dParr, nus, wav, res
    )

    fpfs_spectrum = rprs ** 2 * Fcgs / bb_star_transformed.value

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
    numpyro.factor('spectrum', dist.Normal(loc=interp_depths,
                                           scale=all_depths_errs).log_prob(
        all_depths))

    kepler_thermal_eclipse_depth_err = numpyro.sample(
        "kep_depth_err", dist.Uniform(low=1e-6, high=100e-6)
    )
    numpyro.factor(
        "obs_spectrum_kepler", dist.Normal(
            loc=kepler_thermal_eclipse_depth_model,
            scale=kepler_thermal_eclipse_depth_err
        ).log_prob(kepler_thermal_eclipse_depth_obs)
    )


def run_mcmc(run_title='tmp'):
    print('Start MCMC')
    mcmc = MCMC(
        sampler=NUTS(
            model, dense_mass=True,
        ),
        num_warmup=5,
        num_samples=10,
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

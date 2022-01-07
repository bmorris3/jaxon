import os

import astropy.units as u
import pandas as pd

planets = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'data', 'nea_transiting.csv'),
    index_col='pl_name'
)


def get_planet_params(planet_name):
    planet_props = planets.loc[planet_name]

    t0 = planet_props.t0
    period = planet_props.period
    rstar = planet_props.rstar * u.R_sun
    a_rs = planet_props.a_rstar
    b = planet_props.b
    rho_star = planet_props.rho_star * u.g / u.cm ** 3
    T_s = planet_props.T_s

    a_rp = planet_props.a_rp
    rp_rstar = planet_props.rp_rstar
    eclipse_half_dur = planet_props.half_duration
    mstar = planet_props.mstar
    mass = planet_props.mass
    return (
        planet_name, a_rs, a_rp, T_s, rp_rstar, t0, period, eclipse_half_dur,
        b, rstar.value, rho_star, rp_rstar, mstar, mass
    )
